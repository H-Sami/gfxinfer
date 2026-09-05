// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gfxinfer/status.h"

namespace gfxinfer {

struct Qwen38Spec {
  static constexpr std::uint32_t vocabulary_size = 248320;
  static constexpr std::uint32_t tokenizer_vocabulary_size = 248077;
  static constexpr std::uint32_t hidden_size = 5120;
  static constexpr std::uint32_t intermediate_size = 17408;
  static constexpr std::uint32_t layer_count = 64;
  static constexpr std::uint32_t full_attention_interval = 4;
  static constexpr std::uint32_t attention_head_count = 24;
  static constexpr std::uint32_t kv_head_count = 4;
  static constexpr std::uint32_t attention_head_size = 256;
  static constexpr std::uint32_t rotary_size = 64;
  static constexpr std::uint32_t gdn_key_head_count = 16;
  static constexpr std::uint32_t gdn_key_head_size = 128;
  static constexpr std::uint32_t gdn_value_head_count = 48;
  static constexpr std::uint32_t gdn_value_head_size = 128;
  static constexpr std::uint32_t gdn_convolution_width = 4;
  static constexpr std::uint32_t maximum_context_tokens = 262144;
  static constexpr std::uint32_t mtp_layer_count = 1;
  static constexpr float rms_norm_epsilon = 1.0e-6F;
  static constexpr float rope_theta = 10000000.0F;

  static constexpr std::uint32_t vision_depth = 27;
  static constexpr std::uint32_t vision_hidden_size = 1152;
  static constexpr std::uint32_t vision_intermediate_size = 4304;
  static constexpr std::uint32_t vision_head_count = 16;
  static constexpr std::uint32_t vision_position_count = 2304;
  static constexpr std::uint32_t vision_patch_size = 16;
  static constexpr std::uint32_t vision_temporal_patch_size = 2;
  static constexpr std::uint32_t vision_merge_size = 2;

  [[nodiscard]] static constexpr bool is_full_attention(std::uint32_t layer) {
    return layer < layer_count && (layer + 1U) % full_attention_interval == 0;
  }
};

struct Qwen38SourceTensorSpec {
  std::string name;
  std::vector<std::uint64_t> shape;
};

[[nodiscard]] const std::vector<Qwen38SourceTensorSpec>& qwen38_source_tensor_specs();
[[nodiscard]] const Qwen38SourceTensorSpec* qwen38_source_tensor_spec(
    std::string_view name) noexcept;

[[nodiscard]] std::vector<std::string> qwen38_source_tensor_names(
    bool include_vision = true,
    bool include_mtp = true);

[[nodiscard]] Status validate_qwen38_source_inventory(
    std::span<const std::string> names,
    bool include_vision = true,
    bool include_mtp = true);

}  // namespace gfxinfer
