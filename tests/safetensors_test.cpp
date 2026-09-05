// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

bool safetensors_tests() {
  const auto test_directory = std::filesystem::path(GFXINFER_TEST_OUTPUT_DIR);
  std::error_code ignored;
  std::filesystem::create_directories(test_directory, ignored);
  const auto path = test_directory / "gfxinfer-safetensors-test.safetensors";
  std::string header =
      R"({"weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"__metadata__":{"format":"pt"}})";
  while (header.size() % 8 != 0) header.push_back(' ');
  std::array<std::uint8_t, 8> length{};
  const auto header_bytes = static_cast<std::uint64_t>(header.size());
  for (unsigned byte = 0; byte < 8; ++byte) {
    length[byte] = static_cast<std::uint8_t>(header_bytes >> (byte * 8U));
  }
  const std::array<std::uint8_t, 8> payload{1, 2, 3, 4, 5, 6, 7, 8};
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(length.data()), length.size());
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
  }

  auto file = gfxinfer::SafeTensorFile::open(path);
  bool passed = true;
  if (!file.ok()) {
    std::cerr << "safetensors test: " << file.status().message() << '\n';
    passed = false;
  } else {
    const auto* tensor = file.value().find("weight");
    if (tensor == nullptr || tensor->dtype != gfxinfer::SourceDType::bf16 ||
        tensor->shape != std::vector<std::uint64_t>({2, 2}) || tensor->bytes != 8) {
      std::cerr << "safetensors test: descriptor mismatch\n";
      passed = false;
    } else if (std::to_integer<std::uint8_t>(file.value().data(*tensor)[7]) != 8) {
      std::cerr << "safetensors test: payload mismatch\n";
      passed = false;
    }
  }
  std::filesystem::remove(path, ignored);
  return passed;
}
