// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/converter.h"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gfxinfer/artifact.h"
#include "gfxinfer/quantization.h"
#include "gfxinfer/qwen38_spec.h"
#include "gfxinfer/safetensors.h"

namespace gfxinfer {
namespace {

constexpr std::size_t kDraftHeadRows = 98304;

enum class DerivedKind {
  none,
  draft_head,
  draft_token_ids,
};

struct PlannedItem {
  TensorWritePlan output;
  std::string source_name;
  std::filesystem::path resource_path;
  DerivedKind derived{DerivedKind::none};
  std::vector<std::uint32_t> selected_rows;
};

[[nodiscard]] bool checked_multiply(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t& output) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
  output = left * right;
  return true;
}

[[nodiscard]] Result<std::vector<std::byte>> read_file(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return Status{ErrorCode::io_error, "cannot open resource: " + path.string()};
  }
  const auto end = input.tellg();
  if (end <= 0) {
    return Status{ErrorCode::io_error, "resource is empty or unreadable: " + path.string()};
  }
  const auto size = static_cast<std::uint64_t>(end);
  if (size > std::numeric_limits<std::size_t>::max()) {
    return Status{ErrorCode::unsupported, "resource is too large: " + path.string()};
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    return Status{ErrorCode::io_error, "cannot read resource: " + path.string()};
  }
  return bytes;
}

[[nodiscard]] bool is_early_mlp(
    std::string_view name,
    std::size_t w4_tail_layers,
    std::size_t& layer_out) {
  constexpr std::string_view prefix = "model.language_model.layers.";
  if (!name.starts_with(prefix)) return false;
  const auto layer_end = name.find('.', prefix.size());
  if (layer_end == std::string_view::npos ||
      name.find(".mlp.", layer_end) == std::string_view::npos) {
    return false;
  }
  std::size_t layer = 0;
  for (const char digit : name.substr(prefix.size(), layer_end - prefix.size())) {
    if (digit < '0' || digit > '9') return false;
    layer = layer * 10U + static_cast<std::uint32_t>(digit - '0');
  }
  layer_out = layer;
  return layer < Qwen38Spec::layer_count - w4_tail_layers;
}

[[nodiscard]] Quantization choose_quantization(
    std::string_view name,
    const SafeTensorDescriptor& tensor,
    const Qwen38ConversionOptions& options) {
  if (tensor.shape.size() != 2 || tensor.shape[0] % 16 != 0 || tensor.shape[1] % 128 != 0) {
    return Quantization::none;
  }
  if (name == "lm_head.weight") {
    return options.output_head_bits == 2
               ? Quantization::w2g128
               : (options.output_head_bits == 3 ? Quantization::w3g128
                                                : Quantization::w4g128);
  }
  std::size_t layer = 0;
  if (!is_early_mlp(name, options.w4_tail_layers, layer)) {
    return Quantization::w4g128;
  }
  if (options.mlp_bits == 4) return Quantization::w4g128;
  return options.mlp_bits == 3 && layer >= options.w3_start_layer
             ? Quantization::w3g128
             : Quantization::w2g128;
}

[[nodiscard]] Result<std::vector<std::uint32_t>> select_draft_head_rows(
    const std::filesystem::path& ranking_path) {
  std::ifstream input(ranking_path, std::ios::binary | std::ios::ate);
  if (!input) {
    return Status{ErrorCode::io_error,
                  "cannot open draft-token ranking: " + ranking_path.string()};
  }
  const auto file_bytes = input.tellg();
  constexpr std::size_t row_bytes =
      Qwen38Spec::vocabulary_size * sizeof(std::int64_t);
  if (file_bytes < static_cast<std::streamoff>(row_bytes) ||
      static_cast<std::uint64_t>(file_bytes) % row_bytes != 0) {
    return Status{ErrorCode::corrupt_artifact,
                  "draft-token ranking has an invalid shape"};
  }
  std::vector<std::int64_t> counts(Qwen38Spec::vocabulary_size);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(counts.data()),
             static_cast<std::streamsize>(row_bytes));
  if (!input) {
    return Status{ErrorCode::io_error, "cannot read draft-token ranking"};
  }

  std::vector<std::uint32_t> rows(Qwen38Spec::tokenizer_vocabulary_size);
  std::iota(rows.begin(), rows.end(), 0U);
  const auto better = [&](std::uint32_t left, std::uint32_t right) {
    if (counts[left] != counts[right]) return counts[left] > counts[right];
    return left < right;
  };
  std::partial_sort(rows.begin(), rows.begin() + kDraftHeadRows, rows.end(), better);
  rows.resize(kDraftHeadRows);

  constexpr std::uint32_t required_special_tokens[] = {
      248044, 248045, 248046, 248047, 248048, 248049, 248050,
      248051, 248052, 248053, 248054, 248055, 248056, 248057,
      248070, 248071, 248072, 248073, 248074, 248075, 248076,
  };
  std::vector<bool> selected(Qwen38Spec::tokenizer_vocabulary_size, false);
  for (const auto row : rows) selected[row] = true;
  std::size_t replacement = rows.size();
  for (const auto token : required_special_tokens) {
    if (selected[token]) continue;
    while (replacement != 0) {
      --replacement;
      const auto candidate = rows[replacement];
      if (std::find(std::begin(required_special_tokens),
                    std::end(required_special_tokens), candidate) ==
          std::end(required_special_tokens)) {
        selected[candidate] = false;
        rows[replacement] = token;
        selected[token] = true;
        break;
      }
    }
  }
  std::sort(rows.begin(), rows.end(), better);
  return rows;
}

[[nodiscard]] Result<std::vector<PlannedItem>> make_plan(
    const Qwen38ConversionOptions& options,
    const SafeTensorIndex& index,
    const std::map<std::string, std::size_t>& shard_indices,
    const std::vector<SafeTensorFile>& shards,
    Qwen38ConversionSummary& summary) {
  std::vector<PlannedItem> items;
  const auto names = qwen38_source_tensor_names(false, true);
  items.reserve(names.size() + 9);

  for (const auto& name : names) {
    const auto mapped = index.weight_map.find(name);
    if (mapped == index.weight_map.end()) {
      return Status{ErrorCode::corrupt_artifact, "source index is missing tensor: " + name};
    }
    const auto shard_index = shard_indices.find(mapped->second);
    if (shard_index == shard_indices.end()) {
      return Status{ErrorCode::internal, "source shard was not opened: " + mapped->second};
    }
    const auto* source = shards[shard_index->second].find(name);
    if (source == nullptr || source->dtype != SourceDType::bf16) {
      return Status{ErrorCode::corrupt_artifact, "source BF16 tensor is unavailable: " + name};
    }

    PlannedItem item;
    item.source_name = name;
    item.output.name = name;
    item.output.dimensions = source->shape;
    item.output.alignment = 256;
    item.output.quantization = choose_quantization(name, *source, options);
    if (item.output.quantization == Quantization::none) {
      item.output.dtype = DType::bf16;
      item.output.data_bytes = source->bytes;
      ++summary.bf16_tensor_count;
    } else {
      const auto rows = source->shape[0];
      const auto columns = source->shape[1];
      std::uint64_t code_bytes = 0;
      std::uint64_t scale_count = 0;
      if (!checked_multiply(rows, packed_row_bytes(item.output.quantization, columns),
                            code_bytes) ||
          !checked_multiply(rows, columns / 128, scale_count) ||
          !checked_multiply(scale_count, sizeof(__half), item.output.auxiliary_bytes)) {
        return Status{ErrorCode::unsupported, "quantized tensor size overflow: " + name};
      }
      item.output.dtype = DType::u8;
      item.output.data_bytes = code_bytes;
      if (item.output.quantization == Quantization::w2g128) {
        ++summary.w2_tensor_count;
      } else if (item.output.quantization == Quantization::w3g128) {
        ++summary.w3_tensor_count;
      } else {
        ++summary.w4_tensor_count;
      }
    }
    if (summary.planned_payload_bytes >
        std::numeric_limits<std::uint64_t>::max() - item.output.data_bytes -
            item.output.auxiliary_bytes) {
      return Status{ErrorCode::unsupported, "artifact payload size overflow"};
    }
    summary.planned_payload_bytes +=
        item.output.data_bytes + item.output.auxiliary_bytes;
    items.push_back(std::move(item));
  }
  summary.model_tensor_count = names.size();

  if (!options.draft_ranking_path.empty()) {
    auto selected_rows = select_draft_head_rows(options.draft_ranking_path);
    if (!selected_rows.ok()) return selected_rows.status();
    const auto mapped = index.weight_map.find("lm_head.weight");
    if (mapped == index.weight_map.end()) {
      return Status{ErrorCode::corrupt_artifact,
                    "source index is missing lm_head.weight"};
    }
    const auto shard_index = shard_indices.find(mapped->second);
    if (shard_index == shard_indices.end()) {
      return Status{ErrorCode::internal, "LM-head shard was not opened"};
    }
    const auto* source = shards[shard_index->second].find("lm_head.weight");
    if (source == nullptr || source->shape.size() != 2 ||
        source->shape[0] != Qwen38Spec::vocabulary_size ||
        source->shape[1] != Qwen38Spec::hidden_size) {
      return Status{ErrorCode::corrupt_artifact,
                    "source LM head has an invalid shape"};
    }

    PlannedItem head;
    head.output.name = "mtp.draft_head.weight";
    head.output.dtype = DType::u8;
    head.output.quantization = options.output_head_bits == 2
                                   ? Quantization::w2g128
                                   : (options.output_head_bits == 3
                                          ? Quantization::w3g128
                                          : Quantization::w4g128);
    head.output.dimensions = {kDraftHeadRows, Qwen38Spec::hidden_size};
    head.output.alignment = 256;
    head.output.data_bytes =
        kDraftHeadRows * packed_row_bytes(head.output.quantization,
                                          Qwen38Spec::hidden_size);
    head.output.auxiliary_bytes =
        kDraftHeadRows * (Qwen38Spec::hidden_size / 128) * sizeof(__half);
    head.source_name = "lm_head.weight";
    head.derived = DerivedKind::draft_head;
    head.selected_rows = selected_rows.value();
    summary.planned_payload_bytes +=
        head.output.data_bytes + head.output.auxiliary_bytes;
    if (head.output.quantization == Quantization::w2g128) {
      ++summary.w2_tensor_count;
    } else if (head.output.quantization == Quantization::w3g128) {
      ++summary.w3_tensor_count;
    } else {
      ++summary.w4_tensor_count;
    }
    items.push_back(std::move(head));

    PlannedItem ids;
    ids.output.name = "mtp.draft_head_token_ids";
    ids.output.dtype = DType::i32;
    ids.output.quantization = Quantization::none;
    ids.output.dimensions = {kDraftHeadRows};
    ids.output.alignment = 256;
    ids.output.data_bytes = kDraftHeadRows * sizeof(std::uint32_t);
    ids.derived = DerivedKind::draft_token_ids;
    ids.selected_rows = std::move(selected_rows).value();
    summary.planned_payload_bytes += ids.output.data_bytes;
    items.push_back(std::move(ids));
    summary.model_tensor_count += 2;
  }

  constexpr std::string_view resources[] = {
      "chat_template.jinja",
      "config.json",
      "generation_config.json",
      "merges.txt",
      "preprocessor_config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "video_preprocessor_config.json",
      "vocab.json",
  };
  for (const auto filename : resources) {
    const auto path = options.source_directory / filename;
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error || bytes == 0) {
      return Status{ErrorCode::io_error, "required resource is unavailable: " + path.string()};
    }
    PlannedItem item;
    item.resource_path = path;
    item.output.name = "resources/" + std::string(filename);
    item.output.dtype = DType::u8;
    item.output.quantization = Quantization::none;
    item.output.dimensions = {bytes};
    item.output.data_bytes = bytes;
    summary.planned_payload_bytes += bytes;
    items.push_back(std::move(item));
    ++summary.resource_count;
  }
  return items;
}

[[nodiscard]] float bf16_to_float(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

[[nodiscard]] Status write_quantized(
    ArtifactStreamWriter& writer,
    std::size_t output_index,
    const SafeTensorFile& shard,
    const SafeTensorDescriptor& source,
    Quantization format,
    std::size_t chunk_rows) {
  const auto rows = static_cast<std::size_t>(source.shape[0]);
  const auto columns = static_cast<std::size_t>(source.shape[1]);
  const auto bytes = shard.data(source);
  std::vector<std::uint16_t> bf16;
  std::vector<float> values;
  std::vector<__half> scales;
  for (std::size_t row = 0; row < rows; row += chunk_rows) {
    const auto count = std::min(chunk_rows, rows - row);
    const auto value_count = count * columns;
    bf16.resize(value_count);
    values.resize(value_count);
    std::memcpy(bf16.data(), bytes.data() + row * columns * sizeof(std::uint16_t),
                value_count * sizeof(std::uint16_t));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(value_count); ++index) {
      values[static_cast<std::size_t>(index)] =
          bf16_to_float(bf16[static_cast<std::size_t>(index)]);
    }
    auto packed = quantize_group128(values, count, columns, format);
    if (!packed.ok()) return packed.status();
    auto tiled = tile_group128_for_gfx12(packed.value());
    if (!tiled.ok()) return tiled.status();
    auto status = writer.append_data(
        output_index, std::as_bytes(std::span<const std::uint8_t>(tiled.value().codes)));
    if (!status.ok()) return status;
    scales.resize(tiled.value().scales.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(tiled.value().scales.size()); ++index) {
      scales[static_cast<std::size_t>(index)] =
          __float2half(tiled.value().scales[static_cast<std::size_t>(index)]);
    }
    status = writer.append_auxiliary(
        output_index, std::as_bytes(std::span<const __half>(scales)));
    if (!status.ok()) return status;
  }
  return Status::ok_status();
}

[[nodiscard]] Status write_quantized_rows(
    ArtifactStreamWriter& writer,
    std::size_t output_index,
    const SafeTensorFile& shard,
    const SafeTensorDescriptor& source,
    Quantization format,
    std::span<const std::uint32_t> selected_rows,
    std::size_t chunk_rows) {
  const auto source_rows = static_cast<std::size_t>(source.shape[0]);
  const auto columns = static_cast<std::size_t>(source.shape[1]);
  const auto bytes = shard.data(source);
  std::vector<std::uint16_t> bf16;
  std::vector<float> values;
  std::vector<__half> scales;
  for (std::size_t row = 0; row < selected_rows.size(); row += chunk_rows) {
    const auto count = std::min(chunk_rows, selected_rows.size() - row);
    if (count % 16 != 0) {
      return Status{ErrorCode::invalid_argument,
                    "draft-head conversion chunks must contain a multiple of 16 rows"};
    }
    const auto value_count = count * columns;
    bf16.resize(value_count);
    values.resize(value_count);
    for (std::size_t local = 0; local < count; ++local) {
      const auto source_row = selected_rows[row + local];
      if (source_row >= source_rows) {
        return Status{ErrorCode::corrupt_artifact,
                      "draft-head ranking contains an invalid token ID"};
      }
      std::memcpy(
          bf16.data() + local * columns,
          bytes.data() + static_cast<std::size_t>(source_row) * columns *
                             sizeof(std::uint16_t),
          columns * sizeof(std::uint16_t));
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(value_count); ++index) {
      values[static_cast<std::size_t>(index)] =
          bf16_to_float(bf16[static_cast<std::size_t>(index)]);
    }
    auto packed = quantize_group128(values, count, columns, format);
    if (!packed.ok()) return packed.status();
    auto tiled = tile_group128_for_gfx12(packed.value());
    if (!tiled.ok()) return tiled.status();
    auto status = writer.append_data(
        output_index,
        std::as_bytes(std::span<const std::uint8_t>(tiled.value().codes)));
    if (!status.ok()) return status;
    scales.resize(tiled.value().scales.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(tiled.value().scales.size()); ++index) {
      scales[static_cast<std::size_t>(index)] =
          __float2half(tiled.value().scales[static_cast<std::size_t>(index)]);
    }
    status = writer.append_auxiliary(
        output_index, std::as_bytes(std::span<const __half>(scales)));
    if (!status.ok()) return status;
  }
  return Status::ok_status();
}

}  // namespace

Result<Qwen38ConversionSummary> convert_qwen38(
    const Qwen38ConversionOptions& options,
    ConversionProgress progress) {
  if (options.source_directory.empty() ||
      (!options.plan_only && options.output_path.empty()) ||
      options.chunk_rows == 0 || options.chunk_rows > 4096 ||
      (options.mlp_bits != 2 && options.mlp_bits != 3 && options.mlp_bits != 4) ||
      (options.output_head_bits != 2 && options.output_head_bits != 3 &&
       options.output_head_bits != 4) ||
      options.w3_start_layer > Qwen38Spec::layer_count ||
      options.w4_tail_layers > Qwen38Spec::layer_count ||
      (!options.draft_ranking_path.empty() && options.chunk_rows % 16 != 0)) {
    return Status{ErrorCode::invalid_argument, "invalid Qwen conversion options"};
  }
  auto validation = validate_qwen38_model_directory(options.source_directory, true, true);
  if (!validation.ok()) return validation.status();
  auto index = load_safetensors_index(
      options.source_directory / "model.safetensors.index.json");
  if (!index.ok()) return index.status();

  std::vector<SafeTensorFile> shards;
  shards.reserve(index.value().shards.size());
  std::map<std::string, std::size_t> shard_indices;
  for (const auto& name : index.value().shards) {
    auto shard = SafeTensorFile::open(options.source_directory / name);
    if (!shard.ok()) return shard.status();
    shard_indices.emplace(name, shards.size());
    shards.push_back(std::move(shard).value());
  }

  Qwen38ConversionSummary summary;
  auto plan = make_plan(options, index.value(), shard_indices, shards, summary);
  if (!plan.ok()) return plan.status();
  if (options.plan_only) return summary;

  std::error_code filesystem_error;
  const auto parent = options.output_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, filesystem_error);
    if (filesystem_error) {
      return Status{ErrorCode::io_error,
                    "cannot create artifact directory: " + filesystem_error.message()};
    }
  }
  const auto available = std::filesystem::space(parent.empty() ? "." : parent, filesystem_error);
  if (!filesystem_error && available.available < summary.planned_payload_bytes + (1ULL << 30U)) {
    return Status{ErrorCode::out_of_memory,
                  "insufficient free storage for artifact plus conversion safety margin"};
  }

  std::vector<TensorWritePlan> writer_plan;
  writer_plan.reserve(plan.value().size());
  for (const auto& item : plan.value()) writer_plan.push_back(item.output);
  const bool ranked = !options.draft_ranking_path.empty();
  const bool hybrid = options.mlp_bits == 3 && options.w3_start_layer != 0;
  const bool lowbit_heads = options.output_head_bits != 4;
  ArtifactIdentity identity{
      "gfx1200",
      "qwen3.8-27b",
      lowbit_heads
          ? (options.output_head_bits == 2
                 ? "rdna4-w3seq-w4-h2-a4-t8-f96k-ls"
                 : "rdna4-w3seq-w4-h3-a4-t8-f96k-ls")
          : options.mlp_bits == 4
          ? "rdna4-w4-a4-t8-f96k-ls"
          : options.mlp_bits == 3
          ? (hybrid ? "rdna4-w2-w3seq-w4-a4-t8-f96k-ls"
                    : "rdna4-w3seq-w4-a4-t8-f96k-ls")
          : "rdna4-w2w4-a4-t8",
      lowbit_heads
          ? (options.output_head_bits == 2
                 ? "qwen38-code-w3seq-w4-head2-wmma-tiled-v9"
                 : "qwen38-code-w3seq-w4-head3-wmma-tiled-v9")
          : options.mlp_bits == 4
          ? "qwen38-code-w4-wmma-tiled-v7"
          : options.mlp_bits == 3
          ? (hybrid ? "qwen38-w2-w3seq-w4-wmma-tiled-v9"
                    : "qwen38-code-w3seq-w4-wmma-tiled-v9")
          : (ranked ? "qwen38-maxspeed-wmma-tiled-ranked-v3"
                    : "qwen38-maxspeed-wmma-tiled-v2"),
  };
  auto writer = ArtifactStreamWriter::create(options.output_path, identity, writer_plan);
  if (!writer.ok()) return writer.status();

  for (std::size_t item_index = 0; item_index < plan.value().size(); ++item_index) {
    const auto& item = plan.value()[item_index];
    if (progress) progress(item_index, plan.value().size(), item.output.name);
    Status status;
    if (!item.resource_path.empty()) {
      auto resource = read_file(item.resource_path);
      if (!resource.ok()) return resource.status();
      status = writer.value().append_data(item_index, resource.value());
    } else if (item.derived == DerivedKind::draft_token_ids) {
      status = writer.value().append_data(
          item_index,
          std::as_bytes(std::span<const std::uint32_t>(item.selected_rows)));
    } else {
      const auto mapped = index.value().weight_map.find(item.source_name);
      const auto shard_index = shard_indices.find(mapped->second)->second;
      const auto* source = shards[shard_index].find(item.source_name);
      if (item.output.quantization == Quantization::none) {
        status = writer.value().append_data(item_index, shards[shard_index].data(*source));
      } else if (item.derived == DerivedKind::draft_head) {
        status = write_quantized_rows(
            writer.value(), item_index, shards[shard_index], *source,
            item.output.quantization, item.selected_rows, options.chunk_rows);
      } else {
        status = write_quantized(writer.value(), item_index, shards[shard_index], *source,
                                 item.output.quantization, options.chunk_rows);
      }
    }
    if (!status.ok()) {
      return Status{status.code(), item.output.name + ": " + status.message()};
    }
  }
  if (progress) progress(plan.value().size(), plan.value().size(), "finalizing");
  auto status = writer.value().finalize();
  if (!status.ok()) return status;
  summary.output_file_bytes = std::filesystem::file_size(options.output_path, filesystem_error);
  if (filesystem_error) {
    return Status{ErrorCode::io_error,
                  "cannot stat completed artifact: " + filesystem_error.message()};
  }
  return summary;
}

}  // namespace gfxinfer
