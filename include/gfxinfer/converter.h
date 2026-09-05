// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string_view>

#include "gfxinfer/status.h"

namespace gfxinfer {

struct Qwen38ConversionOptions {
  std::filesystem::path source_directory;
  std::filesystem::path output_path;
  std::filesystem::path draft_ranking_path;
  bool plan_only{false};
  std::size_t chunk_rows{128};
  unsigned mlp_bits{2};
  unsigned output_head_bits{4};
  std::size_t w3_start_layer{0};
  std::size_t w4_tail_layers{8};
};

struct Qwen38ConversionSummary {
  std::size_t model_tensor_count{0};
  std::size_t resource_count{0};
  std::size_t w2_tensor_count{0};
  std::size_t w3_tensor_count{0};
  std::size_t w4_tensor_count{0};
  std::size_t bf16_tensor_count{0};
  std::uint64_t planned_payload_bytes{0};
  std::uint64_t output_file_bytes{0};
};

using ConversionProgress =
    std::function<void(std::size_t completed, std::size_t total, std::string_view name)>;

// Builds the text + one-layer MTP artifact. Vision tensors remain in the source
// checkpoint for the later multimodal artifact profile.
[[nodiscard]] Result<Qwen38ConversionSummary> convert_qwen38(
    const Qwen38ConversionOptions& options,
    ConversionProgress progress = {});

}  // namespace gfxinfer
