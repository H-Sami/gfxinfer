// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/quantization.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool test_format(gfxinfer::Quantization format, float maximum_relative_rmse) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t columns = 256;
  std::mt19937 generator(1200);
  std::normal_distribution<float> distribution(0.0F, 0.25F);
  std::vector<float> values(rows * columns);
  for (auto& value : values) value = distribution(generator);
  values[0] = 0.0F;
  values[columns + 17] = 2.0F;
  values[2 * columns + 93] = -2.0F;

  auto packed = gfxinfer::quantize_group128(values, rows, columns, format);
  if (!packed.ok()) {
    std::cerr << "quantization test: " << packed.status().message() << '\n';
    return false;
  }
  auto restored = gfxinfer::dequantize_group128(packed.value());
  if (!restored.ok()) {
    std::cerr << "quantization test: " << restored.status().message() << '\n';
    return false;
  }
  double squared_error = 0.0;
  double squared_signal = 0.0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    const double difference = values[index] - restored.value()[index];
    squared_error += difference * difference;
    squared_signal += static_cast<double>(values[index]) * values[index];
  }
  const auto relative_rmse = static_cast<float>(std::sqrt(squared_error / squared_signal));
  if (relative_rmse > maximum_relative_rmse) {
    std::cerr << "quantization test: relative RMSE " << relative_rmse
              << " exceeds " << maximum_relative_rmse << '\n';
    return false;
  }
  return true;
}

bool test_gfx12_tiling(gfxinfer::Quantization format) {
  constexpr std::size_t rows = 32;
  constexpr std::size_t columns = 256;
  std::mt19937 generator(1201);
  std::normal_distribution<float> distribution(0.0F, 0.5F);
  std::vector<float> values(rows * columns);
  for (auto& value : values) value = distribution(generator);
  auto packed = gfxinfer::quantize_group128(values, rows, columns, format);
  if (!packed.ok()) return false;
  auto tiled = gfxinfer::tile_group128_for_gfx12(packed.value());
  if (!tiled.ok() || tiled.value().codes.size() != packed.value().codes.size() ||
      tiled.value().scales.size() != packed.value().scales.size()) {
    std::cerr << "quantization test: gfx12 tiling failed\n";
    return false;
  }

  const std::size_t groups = columns / 128;
  const std::size_t fragment_bytes = format == gfxinfer::Quantization::w2g128
                                         ? 8
                                         : (format == gfxinfer::Quantization::w3g128 ? 12 : 16);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const std::size_t tile = row / 16;
      const std::size_t local_row = row % 16;
      const std::size_t group = column / 128;
      const std::size_t k_tile = (column % 128) / 32;
      const std::size_t local_column = column % 32;
      const std::size_t fragment =
          ((((tile * groups + group) * 4 + k_tile) * 16 + local_row) * fragment_bytes);
      int quantized = 0;
      if (format == gfxinfer::Quantization::w3g128) {
        const auto bit = local_column * 3U;
        const auto byte = bit / 8U;
        const auto shift = static_cast<unsigned>(bit & 7U);
        unsigned code = tiled.value().codes[fragment + byte] >> shift;
        if (shift > 5U) {
          code |= static_cast<unsigned>(tiled.value().codes[fragment + byte + 1U])
                  << (8U - shift);
        }
        quantized = static_cast<int>(code & 7U) - 4;
      } else {
        const auto byte = tiled.value().codes[
            fragment + (format == gfxinfer::Quantization::w2g128 ? local_column / 4
                                                                 : local_column / 2)];
        const unsigned shift = static_cast<unsigned>(
            (format == gfxinfer::Quantization::w2g128 ? local_column % 4
                                                      : local_column % 2) *
            (format == gfxinfer::Quantization::w2g128 ? 2 : 4));
        const int code = static_cast<int>(
            (byte >> shift) & (format == gfxinfer::Quantization::w2g128 ? 0x03U : 0x0fU));
        quantized = format == gfxinfer::Quantization::w2g128
                        ? code * 2 - 3
                        : (code >= 8 ? code - 16 : code);
      }
      const float restored = static_cast<float>(quantized) *
                             tiled.value().scales[(tile * groups + group) * 16 + local_row];
      const float expected = gfxinfer::dequantized_weight(packed.value(), row, column);
      if (restored != expected) {
        std::cerr << "quantization test: gfx12 tile mapping mismatch\n";
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool quantization_tests() {
  bool passed = true;
  passed = test_format(gfxinfer::Quantization::w2g128, 0.8F) && passed;
  passed = test_format(gfxinfer::Quantization::w3g128, 0.4F) && passed;
  passed = test_format(gfxinfer::Quantization::w4g128, 0.2F) && passed;
  passed = test_gfx12_tiling(gfxinfer::Quantization::w2g128) && passed;
  passed = test_gfx12_tiling(gfxinfer::Quantization::w3g128) && passed;
  passed = test_gfx12_tiling(gfxinfer::Quantization::w4g128) && passed;

  std::vector<float> invalid(127, 0.0F);
  auto result = gfxinfer::quantize_group128(
      invalid, 1, 127, gfxinfer::Quantization::w4g128);
  if (result.ok()) {
    std::cerr << "quantization test: invalid group size was accepted\n";
    passed = false;
  }
  return passed;
}
