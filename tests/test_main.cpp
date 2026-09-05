// SPDX-License-Identifier: Apache-2.0
#include <iostream>

bool artifact_tests();
bool quantization_tests();
bool qwen38_spec_tests();
bool safetensors_tests();

int main() {
  bool passed = true;
  passed = artifact_tests() && passed;
  passed = quantization_tests() && passed;
  passed = qwen38_spec_tests() && passed;
  passed = safetensors_tests() && passed;
  if (!passed) {
    std::cerr << "GFXInfer tests failed\n";
    return 1;
  }
  std::cout << "GFXInfer tests passed\n";
  return 0;
}
