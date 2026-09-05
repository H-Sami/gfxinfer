// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/qwen38_spec.h"

#include <algorithm>
#include <iostream>

bool qwen38_spec_tests() {
  bool passed = true;
  std::size_t attention_layers = 0;
  for (std::uint32_t layer = 0; layer < gfxinfer::Qwen38Spec::layer_count; ++layer) {
    attention_layers += gfxinfer::Qwen38Spec::is_full_attention(layer) ? 1U : 0U;
  }
  if (attention_layers != 16) {
    std::cerr << "qwen spec test: expected 16 full-attention layers\n";
    passed = false;
  }
  auto text = gfxinfer::qwen38_source_tensor_names(false, false);
  auto complete = gfxinfer::qwen38_source_tensor_names(true, true);
  if (text.size() != 851 || complete.size() != 1199) {
    std::cerr << "qwen spec test: unexpected inventory counts text=" << text.size()
              << " complete=" << complete.size() << '\n';
    passed = false;
  }
  const auto* q_projection = gfxinfer::qwen38_source_tensor_spec(
      "model.language_model.layers.3.self_attn.q_proj.weight");
  if (q_projection == nullptr ||
      q_projection->shape != std::vector<std::uint64_t>({12288, 5120}) ||
      gfxinfer::qwen38_source_tensor_specs().size() != 1199) {
    std::cerr << "qwen spec test: exact shape registry mismatch\n";
    passed = false;
  }
  if (!std::binary_search(complete.begin(), complete.end(),
                          "model.language_model.layers.63.self_attn.q_proj.weight") ||
      !std::binary_search(complete.begin(), complete.end(), "mtp.fc.weight") ||
      !std::binary_search(complete.begin(), complete.end(),
                          "model.visual.blocks.26.attn.qkv.weight")) {
    std::cerr << "qwen spec test: required source name missing\n";
    passed = false;
  }
  auto status = gfxinfer::validate_qwen38_source_inventory(complete, true, true);
  if (!status.ok()) {
    std::cerr << "qwen spec test: self-validation failed: " << status.message() << '\n';
    passed = false;
  }
  complete.pop_back();
  if (gfxinfer::validate_qwen38_source_inventory(complete, true, true).ok()) {
    std::cerr << "qwen spec test: incomplete inventory accepted\n";
    passed = false;
  }
  return passed;
}
