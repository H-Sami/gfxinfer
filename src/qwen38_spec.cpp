// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/qwen38_spec.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <string_view>

namespace gfxinfer {
namespace {

void append(std::vector<std::string>& names, const std::string& prefix,
            std::initializer_list<std::string_view> suffixes) {
  for (const auto suffix : suffixes) {
    names.push_back(prefix + std::string(suffix));
  }
}

void add_spec(
    std::vector<Qwen38SourceTensorSpec>& specs,
    std::string name,
    std::initializer_list<std::uint64_t> shape) {
  specs.push_back({std::move(name), std::vector<std::uint64_t>(shape)});
}

}  // namespace

const std::vector<Qwen38SourceTensorSpec>& qwen38_source_tensor_specs() {
  static const auto specs = [] {
    std::vector<Qwen38SourceTensorSpec> output;
    output.reserve(1199);
    add_spec(output, "lm_head.weight", {248320, 5120});
    add_spec(output, "model.language_model.embed_tokens.weight", {248320, 5120});
    add_spec(output, "model.language_model.norm.weight", {5120});

    for (std::uint32_t layer = 0; layer < Qwen38Spec::layer_count; ++layer) {
      const auto prefix = "model.language_model.layers." + std::to_string(layer) + ".";
      add_spec(output, prefix + "input_layernorm.weight", {5120});
      add_spec(output, prefix + "mlp.down_proj.weight", {5120, 17408});
      add_spec(output, prefix + "mlp.gate_proj.weight", {17408, 5120});
      add_spec(output, prefix + "mlp.up_proj.weight", {17408, 5120});
      add_spec(output, prefix + "post_attention_layernorm.weight", {5120});
      if (Qwen38Spec::is_full_attention(layer)) {
        add_spec(output, prefix + "self_attn.k_norm.weight", {256});
        add_spec(output, prefix + "self_attn.k_proj.weight", {1024, 5120});
        add_spec(output, prefix + "self_attn.o_proj.weight", {5120, 6144});
        add_spec(output, prefix + "self_attn.q_norm.weight", {256});
        add_spec(output, prefix + "self_attn.q_proj.weight", {12288, 5120});
        add_spec(output, prefix + "self_attn.v_proj.weight", {1024, 5120});
      } else {
        add_spec(output, prefix + "linear_attn.A_log", {48});
        add_spec(output, prefix + "linear_attn.conv1d.weight", {10240, 1, 4});
        add_spec(output, prefix + "linear_attn.dt_bias", {48});
        add_spec(output, prefix + "linear_attn.in_proj_a.weight", {48, 5120});
        add_spec(output, prefix + "linear_attn.in_proj_b.weight", {48, 5120});
        add_spec(output, prefix + "linear_attn.in_proj_qkv.weight", {10240, 5120});
        add_spec(output, prefix + "linear_attn.in_proj_z.weight", {6144, 5120});
        add_spec(output, prefix + "linear_attn.norm.weight", {128});
        add_spec(output, prefix + "linear_attn.out_proj.weight", {5120, 6144});
      }
    }

    for (std::uint32_t block = 0; block < Qwen38Spec::vision_depth; ++block) {
      const auto prefix = "model.visual.blocks." + std::to_string(block) + ".";
      add_spec(output, prefix + "attn.proj.bias", {1152});
      add_spec(output, prefix + "attn.proj.weight", {1152, 1152});
      add_spec(output, prefix + "attn.qkv.bias", {3456});
      add_spec(output, prefix + "attn.qkv.weight", {3456, 1152});
      add_spec(output, prefix + "mlp.linear_fc1.bias", {4304});
      add_spec(output, prefix + "mlp.linear_fc1.weight", {4304, 1152});
      add_spec(output, prefix + "mlp.linear_fc2.bias", {1152});
      add_spec(output, prefix + "mlp.linear_fc2.weight", {1152, 4304});
      add_spec(output, prefix + "norm1.bias", {1152});
      add_spec(output, prefix + "norm1.weight", {1152});
      add_spec(output, prefix + "norm2.bias", {1152});
      add_spec(output, prefix + "norm2.weight", {1152});
    }
    add_spec(output, "model.visual.merger.linear_fc1.bias", {4608});
    add_spec(output, "model.visual.merger.linear_fc1.weight", {4608, 4608});
    add_spec(output, "model.visual.merger.linear_fc2.bias", {5120});
    add_spec(output, "model.visual.merger.linear_fc2.weight", {5120, 4608});
    add_spec(output, "model.visual.merger.norm.bias", {1152});
    add_spec(output, "model.visual.merger.norm.weight", {1152});
    add_spec(output, "model.visual.patch_embed.proj.bias", {1152});
    add_spec(output, "model.visual.patch_embed.proj.weight", {1152, 3, 2, 16, 16});
    add_spec(output, "model.visual.pos_embed.weight", {2304, 1152});

    add_spec(output, "mtp.fc.weight", {5120, 10240});
    add_spec(output, "mtp.layers.0.input_layernorm.weight", {5120});
    add_spec(output, "mtp.layers.0.mlp.down_proj.weight", {5120, 17408});
    add_spec(output, "mtp.layers.0.mlp.gate_proj.weight", {17408, 5120});
    add_spec(output, "mtp.layers.0.mlp.up_proj.weight", {17408, 5120});
    add_spec(output, "mtp.layers.0.post_attention_layernorm.weight", {5120});
    add_spec(output, "mtp.layers.0.self_attn.k_norm.weight", {256});
    add_spec(output, "mtp.layers.0.self_attn.k_proj.weight", {1024, 5120});
    add_spec(output, "mtp.layers.0.self_attn.o_proj.weight", {5120, 6144});
    add_spec(output, "mtp.layers.0.self_attn.q_norm.weight", {256});
    add_spec(output, "mtp.layers.0.self_attn.q_proj.weight", {12288, 5120});
    add_spec(output, "mtp.layers.0.self_attn.v_proj.weight", {1024, 5120});
    add_spec(output, "mtp.norm.weight", {5120});
    add_spec(output, "mtp.pre_fc_norm_embedding.weight", {5120});
    add_spec(output, "mtp.pre_fc_norm_hidden.weight", {5120});

    std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
      return left.name < right.name;
    });
    return output;
  }();
  return specs;
}

const Qwen38SourceTensorSpec* qwen38_source_tensor_spec(std::string_view name) noexcept {
  const auto& specs = qwen38_source_tensor_specs();
  const auto found = std::lower_bound(
      specs.begin(), specs.end(), name,
      [](const auto& spec, std::string_view target) { return spec.name < target; });
  return found != specs.end() && found->name == name ? &*found : nullptr;
}

std::vector<std::string> qwen38_source_tensor_names(bool include_vision, bool include_mtp) {
  std::vector<std::string> names;
  names.reserve(1199);
  names.emplace_back("lm_head.weight");
  names.emplace_back("model.language_model.embed_tokens.weight");
  names.emplace_back("model.language_model.norm.weight");

  for (std::uint32_t layer = 0; layer < Qwen38Spec::layer_count; ++layer) {
    const auto prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    append(names, prefix, {
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
    });
    if (Qwen38Spec::is_full_attention(layer)) {
      append(names, prefix, {
          "self_attn.k_norm.weight",
          "self_attn.k_proj.weight",
          "self_attn.o_proj.weight",
          "self_attn.q_norm.weight",
          "self_attn.q_proj.weight",
          "self_attn.v_proj.weight",
      });
    } else {
      append(names, prefix, {
          "linear_attn.A_log",
          "linear_attn.conv1d.weight",
          "linear_attn.dt_bias",
          "linear_attn.in_proj_a.weight",
          "linear_attn.in_proj_b.weight",
          "linear_attn.in_proj_qkv.weight",
          "linear_attn.in_proj_z.weight",
          "linear_attn.norm.weight",
          "linear_attn.out_proj.weight",
      });
    }
  }

  if (include_vision) {
    for (std::uint32_t block = 0; block < Qwen38Spec::vision_depth; ++block) {
      const auto prefix = "model.visual.blocks." + std::to_string(block) + ".";
      append(names, prefix, {
          "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
          "mlp.linear_fc1.bias", "mlp.linear_fc1.weight",
          "mlp.linear_fc2.bias", "mlp.linear_fc2.weight",
          "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
      });
    }
    append(names, "model.visual.", {
        "merger.linear_fc1.bias", "merger.linear_fc1.weight",
        "merger.linear_fc2.bias", "merger.linear_fc2.weight",
        "merger.norm.bias", "merger.norm.weight",
        "patch_embed.proj.bias", "patch_embed.proj.weight", "pos_embed.weight",
    });
  }

  if (include_mtp) {
    append(names, "mtp.", {
        "fc.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "norm.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    });
  }
  std::sort(names.begin(), names.end());
  return names;
}

Status validate_qwen38_source_inventory(
    std::span<const std::string> names,
    bool include_vision,
    bool include_mtp) {
  const auto expected_vector = qwen38_source_tensor_names(include_vision, include_mtp);
  const std::set<std::string> expected(expected_vector.begin(), expected_vector.end());
  const std::set<std::string> actual(names.begin(), names.end());
  if (actual.size() != names.size()) {
    return {ErrorCode::invalid_argument, "source inventory contains duplicate tensor names"};
  }
  if (actual == expected) {
    return Status::ok_status();
  }

  std::ostringstream message;
  message << "source inventory mismatch";
  std::size_t reported = 0;
  for (const auto& name : expected) {
    if (!actual.contains(name) && reported++ < 8) message << "\n  missing: " << name;
  }
  reported = 0;
  for (const auto& name : actual) {
    if (!expected.contains(name) && reported++ < 8) message << "\n  unexpected: " << name;
  }
  return {ErrorCode::invalid_argument, message.str()};
}

}  // namespace gfxinfer
