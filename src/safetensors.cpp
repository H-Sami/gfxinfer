// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/safetensors.h"

#include <simdjson.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "gfxinfer/qwen38_spec.h"

namespace gfxinfer {
namespace {

[[nodiscard]] Status io_error(std::string_view action, const std::filesystem::path& path) {
  return {ErrorCode::io_error,
          std::string(action) + " " + path.string() + ": " + std::strerror(errno)};
}

[[nodiscard]] Status json_error(std::string_view context, simdjson::error_code error) {
  return {ErrorCode::corrupt_artifact,
          std::string(context) + ": " + simdjson::error_message(error)};
}

[[nodiscard]] Result<SourceDType> parse_dtype(std::string_view value) {
  if (value == "BF16") return SourceDType::bf16;
  if (value == "F16") return SourceDType::f16;
  if (value == "F32") return SourceDType::f32;
  if (value == "U8") return SourceDType::u8;
  if (value == "I8") return SourceDType::i8;
  if (value == "I32") return SourceDType::i32;
  if (value == "I64") return SourceDType::i64;
  return Status{ErrorCode::unsupported, "unsupported safetensors dtype: " + std::string(value)};
}

[[nodiscard]] bool safe_product(
    std::span<const std::uint64_t> shape,
    std::uint64_t element_bytes,
    std::uint64_t& output) {
  output = element_bytes;
  for (const auto dimension : shape) {
    if (dimension != 0 && output > std::numeric_limits<std::uint64_t>::max() / dimension) {
      return false;
    }
    output *= dimension;
  }
  return true;
}

[[nodiscard]] bool range_valid(
    std::uint64_t offset,
    std::uint64_t bytes,
    std::uint64_t file_bytes) {
  return offset <= file_bytes && bytes <= file_bytes - offset;
}

[[nodiscard]] bool safe_shard_name(std::string_view name) {
  if (name.empty() || name.find('/') != std::string_view::npos ||
      name.find("..") != std::string_view::npos) {
    return false;
  }
  return name.ends_with(".safetensors");
}

}  // namespace

const char* source_dtype_name(SourceDType dtype) noexcept {
  switch (dtype) {
    case SourceDType::bf16: return "BF16";
    case SourceDType::f16: return "F16";
    case SourceDType::f32: return "F32";
    case SourceDType::u8: return "U8";
    case SourceDType::i8: return "I8";
    case SourceDType::i32: return "I32";
    case SourceDType::i64: return "I64";
  }
  return "unknown";
}

std::size_t source_dtype_bytes(SourceDType dtype) noexcept {
  switch (dtype) {
    case SourceDType::bf16:
    case SourceDType::f16: return 2;
    case SourceDType::f32:
    case SourceDType::i32: return 4;
    case SourceDType::u8:
    case SourceDType::i8: return 1;
    case SourceDType::i64: return 8;
  }
  return 0;
}

SafeTensorFile::~SafeTensorFile() { reset(); }

SafeTensorFile::SafeTensorFile(SafeTensorFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      mapped_bytes_(std::exchange(other.mapped_bytes_, 0)),
      tensors_(std::move(other.tensors_)) {}

SafeTensorFile& SafeTensorFile::operator=(SafeTensorFile&& other) noexcept {
  if (this != &other) {
    reset();
    fd_ = std::exchange(other.fd_, -1);
    mapping_ = std::exchange(other.mapping_, nullptr);
    mapped_bytes_ = std::exchange(other.mapped_bytes_, 0);
    tensors_ = std::move(other.tensors_);
  }
  return *this;
}

void SafeTensorFile::reset() noexcept {
  if (mapping_ != nullptr) {
    (void)::munmap(const_cast<std::byte*>(mapping_), mapped_bytes_);
  }
  if (fd_ >= 0) {
    (void)::close(fd_);
  }
  fd_ = -1;
  mapping_ = nullptr;
  mapped_bytes_ = 0;
  tensors_.clear();
}

Result<SafeTensorFile> SafeTensorFile::open(const std::filesystem::path& path) {
  SafeTensorFile file;
  file.fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file.fd_ < 0) return io_error("cannot open", path);

  struct stat file_stat {};
  if (::fstat(file.fd_, &file_stat) != 0) return io_error("cannot stat", path);
  if (file_stat.st_size < 10) {
    return Status{ErrorCode::corrupt_artifact, "safetensors file is too small"};
  }
  file.mapped_bytes_ = static_cast<std::size_t>(file_stat.st_size);
  const auto* mapping = ::mmap(nullptr, file.mapped_bytes_, PROT_READ, MAP_PRIVATE, file.fd_, 0);
  if (mapping == MAP_FAILED) {
    file.mapping_ = nullptr;
    return io_error("cannot map", path);
  }
  file.mapping_ = static_cast<const std::byte*>(mapping);

  std::uint64_t header_bytes = 0;
  for (unsigned byte = 0; byte < 8; ++byte) {
    header_bytes |= static_cast<std::uint64_t>(
                        std::to_integer<std::uint8_t>(file.mapping_[byte]))
                    << (byte * 8U);
  }
  constexpr std::uint64_t maximum_header_bytes = 128ULL * 1024ULL * 1024ULL;
  if (header_bytes < 2 || header_bytes > maximum_header_bytes ||
      !range_valid(8, header_bytes, file.mapped_bytes_)) {
    return Status{ErrorCode::corrupt_artifact, "invalid safetensors header length"};
  }

  const std::string header(reinterpret_cast<const char*>(file.mapping_ + 8),
                           static_cast<std::size_t>(header_bytes));
  simdjson::dom::parser parser;
  simdjson::dom::element root;
  if (const auto error = parser.parse(simdjson::padded_string(header)).get(root); error) {
    return json_error("invalid safetensors header JSON", error);
  }
  simdjson::dom::object object;
  if (const auto error = root.get(object); error) {
    return json_error("safetensors header must be an object", error);
  }

  const std::uint64_t data_base = 8 + header_bytes;
  std::set<std::string> names;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  for (const auto field : object) {
    const std::string name(field.key);
    if (name == "__metadata__") continue;
    if (!names.insert(name).second) {
      return Status{ErrorCode::corrupt_artifact, "duplicate safetensors name: " + name};
    }
    simdjson::dom::object descriptor;
    if (const auto error = field.value.get(descriptor); error) {
      return json_error("tensor descriptor must be an object", error);
    }
    std::string_view dtype_text;
    if (const auto error = descriptor["dtype"].get(dtype_text); error) {
      return json_error("tensor dtype is missing or invalid", error);
    }
    auto dtype = parse_dtype(dtype_text);
    if (!dtype.ok()) return dtype.status();

    simdjson::dom::array shape_json;
    if (const auto error = descriptor["shape"].get(shape_json); error) {
      return json_error("tensor shape is missing or invalid", error);
    }
    std::vector<std::uint64_t> shape;
    for (const auto dimension_json : shape_json) {
      std::uint64_t dimension = 0;
      if (const auto error = dimension_json.get(dimension); error) {
        return json_error("tensor shape contains a non-integer", error);
      }
      shape.push_back(dimension);
    }

    simdjson::dom::array offsets_json;
    if (const auto error = descriptor["data_offsets"].get(offsets_json); error) {
      return json_error("tensor data_offsets are missing or invalid", error);
    }
    std::vector<std::uint64_t> offsets;
    for (const auto offset_json : offsets_json) {
      std::uint64_t offset = 0;
      if (const auto error = offset_json.get(offset); error) {
        return json_error("tensor offset is not an integer", error);
      }
      offsets.push_back(offset);
    }
    if (offsets.size() != 2 || offsets[1] < offsets[0]) {
      return Status{ErrorCode::corrupt_artifact, "invalid tensor offsets: " + name};
    }
    std::uint64_t expected_bytes = 0;
    if (!safe_product(shape, source_dtype_bytes(dtype.value()), expected_bytes) ||
        expected_bytes != offsets[1] - offsets[0] ||
        !range_valid(data_base + offsets[0], expected_bytes, file.mapped_bytes_)) {
      return Status{ErrorCode::corrupt_artifact,
                    "tensor shape, dtype, and byte range disagree: " + name};
    }
    file.tensors_.push_back({name, dtype.value(), std::move(shape),
                             data_base + offsets[0], expected_bytes});
    ranges.emplace_back(offsets[0], offsets[1]);
  }

  std::sort(file.tensors_.begin(), file.tensors_.end(),
            [](const auto& left, const auto& right) { return left.name < right.name; });
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t index = 1; index < ranges.size(); ++index) {
    if (ranges[index].first < ranges[index - 1].second) {
      return Status{ErrorCode::corrupt_artifact, "safetensors data ranges overlap"};
    }
  }
  return Result<SafeTensorFile>(std::move(file));
}

const SafeTensorDescriptor* SafeTensorFile::find(std::string_view name) const noexcept {
  const auto found = std::lower_bound(
      tensors_.begin(), tensors_.end(), name,
      [](const auto& tensor, std::string_view target) { return tensor.name < target; });
  return found != tensors_.end() && found->name == name ? &*found : nullptr;
}

std::span<const std::byte> SafeTensorFile::data(const SafeTensorDescriptor& tensor) const {
  return {mapping_ + tensor.file_offset, static_cast<std::size_t>(tensor.bytes)};
}

Result<SafeTensorIndex> load_safetensors_index(const std::filesystem::path& path) {
  simdjson::dom::parser parser;
  simdjson::dom::element root;
  if (const auto error = parser.load(path.string()).get(root); error) {
    return json_error("cannot parse safetensors index", error);
  }
  simdjson::dom::object object;
  if (const auto error = root.get(object); error) return json_error("index root", error);

  SafeTensorIndex index;
  double total_size = 0.0;
  if (const auto error = object["metadata"]["total_size"].get(total_size); error ||
      total_size < 0.0 || total_size > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
    return Status{ErrorCode::corrupt_artifact, "index metadata.total_size is invalid"};
  }
  index.tensor_bytes = static_cast<std::uint64_t>(total_size);

  simdjson::dom::object weight_map;
  if (const auto error = object["weight_map"].get(weight_map); error) {
    return json_error("index weight_map is invalid", error);
  }
  std::set<std::string> shards;
  for (const auto field : weight_map) {
    std::string_view shard;
    if (const auto error = field.value.get(shard); error) {
      return json_error("weight_map shard value is invalid", error);
    }
    if (!safe_shard_name(shard)) {
      return Status{ErrorCode::corrupt_artifact, "unsafe shard name in index: " + std::string(shard)};
    }
    const auto [_, inserted] = index.weight_map.emplace(std::string(field.key), std::string(shard));
    if (!inserted) {
      return Status{ErrorCode::corrupt_artifact,
                    "duplicate tensor in index: " + std::string(field.key)};
    }
    shards.emplace(shard);
  }
  index.shards.assign(shards.begin(), shards.end());
  return index;
}

Result<SourceValidationSummary> validate_qwen38_model_directory(
    const std::filesystem::path& directory,
    bool include_vision,
    bool include_mtp) {
  auto index = load_safetensors_index(directory / "model.safetensors.index.json");
  if (!index.ok()) return index.status();

  std::vector<std::string> indexed_names;
  indexed_names.reserve(index.value().weight_map.size());
  for (const auto& [name, _] : index.value().weight_map) indexed_names.push_back(name);
  auto inventory_status = validate_qwen38_source_inventory(
      indexed_names, include_vision, include_mtp);
  if (!inventory_status.ok()) return inventory_status;

  std::set<std::string> observed;
  std::uint64_t observed_bytes = 0;
  for (const auto& shard_name : index.value().shards) {
    auto shard = SafeTensorFile::open(directory / shard_name);
    if (!shard.ok()) return shard.status();
    for (const auto& tensor : shard.value().tensors()) {
      const auto mapped = index.value().weight_map.find(tensor.name);
      if (mapped == index.value().weight_map.end() || mapped->second != shard_name) {
        return Status{ErrorCode::corrupt_artifact,
                      "shard tensor is absent or mapped to a different shard: " + tensor.name};
      }
      if (tensor.dtype != SourceDType::bf16) {
        return Status{ErrorCode::unsupported,
                      "official source tensor is not BF16: " + tensor.name};
      }
      const auto* expected = qwen38_source_tensor_spec(tensor.name);
      if (expected == nullptr || expected->shape != tensor.shape) {
        return Status{ErrorCode::corrupt_artifact,
                      "official source tensor has an unexpected shape: " + tensor.name};
      }
      if (!observed.insert(tensor.name).second) {
        return Status{ErrorCode::corrupt_artifact,
                      "tensor occurs in multiple shards: " + tensor.name};
      }
      observed_bytes += tensor.bytes;
    }
  }
  if (observed.size() != index.value().weight_map.size() ||
      observed_bytes != index.value().tensor_bytes) {
    return Status{ErrorCode::corrupt_artifact,
                  "shard contents do not match index tensor count or total_size"};
  }
  return SourceValidationSummary{observed.size(), index.value().shards.size(), observed_bytes};
}

}  // namespace gfxinfer
