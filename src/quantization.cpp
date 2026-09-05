// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/quantization.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>

namespace gfxinfer {
namespace {

constexpr std::size_t kGroupSize = 128;

[[nodiscard]] int nearest_w2(float normalized) {
  constexpr std::array<int, 4> levels{-3, -1, 1, 3};
  int best = levels.front();
  float best_error = std::numeric_limits<float>::infinity();
  for (const int level : levels) {
    const float error = std::abs(normalized - static_cast<float>(level));
    if (error < best_error) {
      best = level;
      best_error = error;
    }
  }
  return best;
}

[[nodiscard]] std::uint8_t encode_value(float value, float scale, Quantization format) {
  const float normalized = scale == 0.0F ? 0.0F : value / scale;
  switch (format) {
    case Quantization::w2g128: {
      const int quantized = nearest_w2(normalized);
      return static_cast<std::uint8_t>((quantized + 3) / 2);
    }
    case Quantization::w3g128: {
      const int quantized = std::clamp(static_cast<int>(std::nearbyint(normalized)), -4, 3);
      return static_cast<std::uint8_t>(quantized + 4);
    }
    case Quantization::w4g128: {
      const int quantized = std::clamp(static_cast<int>(std::nearbyint(normalized)), -8, 7);
      return static_cast<std::uint8_t>(quantized & 0x0f);
    }
    default: return 0;
  }
}

[[nodiscard]] int decode_value(std::uint8_t code, Quantization format) {
  switch (format) {
    case Quantization::w2g128: return static_cast<int>(code) * 2 - 3;
    case Quantization::w3g128: return static_cast<int>(code) - 4;
    case Quantization::w4g128:
      return (code & 0x08U) != 0 ? static_cast<int>(code) - 16 : static_cast<int>(code);
    default: return 0;
  }
}

[[nodiscard]] float divisor(Quantization format) {
  switch (format) {
    case Quantization::w2g128: return 3.0F;
    case Quantization::w3g128: return 4.0F;
    case Quantization::w4g128: return 8.0F;
    default: return 1.0F;
  }
}

[[nodiscard]] float choose_scale(
    std::span<const float> values,
    float maximum,
    Quantization format) {
  if (maximum == 0.0F) {
    return 1.0F;
  }
  const float base = maximum / divisor(format);
  // Fit the scale against the actual fixed codebook. This matters for W2's
  // zero-less levels and also avoids wasting the positive endpoint in the
  // asymmetric signed W3/W4 codebooks.
  float scale = base;
  for (int iteration = 0; iteration < 4; ++iteration) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (const float value : values) {
      const auto code = encode_value(value, scale, format);
      const int level = decode_value(code, format);
      numerator += static_cast<double>(value) * level;
      denominator += static_cast<double>(level) * level;
    }
    if (denominator == 0.0) break;
    scale = static_cast<float>(numerator / denominator);
  }
  return std::max(scale, std::numeric_limits<float>::min());
}

void store_code(
    std::span<std::uint8_t> row,
    std::size_t column,
    std::uint8_t code,
    Quantization format) {
  switch (format) {
    case Quantization::w2g128: {
      const auto shift = static_cast<unsigned>((column & 3U) * 2U);
      row[column / 4] |= static_cast<std::uint8_t>((code & 0x03U) << shift);
      break;
    }
    case Quantization::w3g128: {
      const std::size_t group = column / kGroupSize;
      const std::size_t local = column % kGroupSize;
      const std::size_t byte = local / 8;
      const auto bit = static_cast<unsigned>(local & 7U);
      const std::size_t group_base = group * 48;
      for (std::size_t plane = 0; plane < 3; ++plane) {
        row[group_base + plane * 16 + byte] |= static_cast<std::uint8_t>(
            ((code >> plane) & 1U) << bit);
      }
      break;
    }
    case Quantization::w4g128: {
      const auto shift = static_cast<unsigned>((column & 1U) * 4U);
      row[column / 2] |= static_cast<std::uint8_t>((code & 0x0fU) << shift);
      break;
    }
    default: break;
  }
}

[[nodiscard]] std::uint8_t load_code(
    std::span<const std::uint8_t> row,
    std::size_t column,
    Quantization format) {
  switch (format) {
    case Quantization::w2g128:
      return static_cast<std::uint8_t>((row[column / 4] >> ((column & 3U) * 2U)) & 0x03U);
    case Quantization::w3g128: {
      const std::size_t group = column / kGroupSize;
      const std::size_t local = column % kGroupSize;
      const std::size_t byte = local / 8;
      const auto bit = static_cast<unsigned>(local & 7U);
      const std::size_t group_base = group * 48;
      std::uint8_t code = 0;
      for (std::size_t plane = 0; plane < 3; ++plane) {
        code |= static_cast<std::uint8_t>(
            ((row[group_base + plane * 16 + byte] >> bit) & 1U) << plane);
      }
      return code;
    }
    case Quantization::w4g128:
      return static_cast<std::uint8_t>((row[column / 2] >> ((column & 1U) * 4U)) & 0x0fU);
    default: return 0;
  }
}

}  // namespace

std::size_t packed_row_bytes(Quantization format, std::size_t columns) {
  switch (format) {
    case Quantization::w2g128: return (columns + 3) / 4;
    case Quantization::w3g128: return (columns * 3 + 7) / 8;
    case Quantization::w4g128: return (columns + 1) / 2;
    default: return 0;
  }
}

Result<PackedMatrix> quantize_group128(
    std::span<const float> weights,
    std::size_t rows,
    std::size_t columns,
    Quantization format) {
  if (rows == 0 || columns == 0 || columns % kGroupSize != 0 ||
      weights.size() != rows * columns) {
    return Status{ErrorCode::invalid_argument,
                  "group-128 quantization requires nonzero dimensions, K divisible by 128, "
                  "and an exact input span"};
  }
  if (format != Quantization::w2g128 && format != Quantization::w3g128 &&
      format != Quantization::w4g128) {
    return Status{ErrorCode::unsupported, "only W2G128, W3G128, and W4G128 are implemented"};
  }

  PackedMatrix output;
  output.rows = rows;
  output.columns = columns;
  output.format = format;
  const auto row_bytes = packed_row_bytes(format, columns);
  output.codes.assign(rows * row_bytes, 0);
  output.scales.resize(rows * (columns / kGroupSize));

  std::atomic<bool> non_finite{false};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t signed_row = 0; signed_row < static_cast<std::int64_t>(rows);
       ++signed_row) {
    const auto row_index = static_cast<std::size_t>(signed_row);
    const auto source = weights.subspan(row_index * columns, columns);
    auto destination = std::span<std::uint8_t>(output.codes).subspan(
        row_index * row_bytes, row_bytes);
    for (std::size_t group = 0; group < columns / kGroupSize; ++group) {
      float maximum = 0.0F;
      for (std::size_t offset = 0; offset < kGroupSize; ++offset) {
        const auto value = source[group * kGroupSize + offset];
        if (!std::isfinite(value)) {
          non_finite.store(true, std::memory_order_relaxed);
          continue;
        }
        maximum = std::max(maximum, std::abs(value));
      }
      const auto group_values = source.subspan(group * kGroupSize, kGroupSize);
      const float scale = choose_scale(group_values, maximum, format);
      output.scales[row_index * (columns / kGroupSize) + group] = scale;
      for (std::size_t offset = 0; offset < kGroupSize; ++offset) {
        const auto column = group * kGroupSize + offset;
        store_code(destination, column, encode_value(source[column], scale, format), format);
      }
    }
  }
  if (non_finite.load(std::memory_order_relaxed)) {
    return Status{ErrorCode::invalid_argument, "weight matrix contains a non-finite value"};
  }
  return output;
}

float dequantized_weight(
    const PackedMatrix& matrix,
    std::size_t row,
    std::size_t column) {
  if (row >= matrix.rows || column >= matrix.columns || matrix.columns % kGroupSize != 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const auto row_bytes = packed_row_bytes(matrix.format, matrix.columns);
  const auto codes = std::span<const std::uint8_t>(matrix.codes).subspan(
      row * row_bytes, row_bytes);
  const auto code = load_code(codes, column, matrix.format);
  const auto scale = matrix.scales[row * (matrix.columns / kGroupSize) + column / kGroupSize];
  return static_cast<float>(decode_value(code, matrix.format)) * scale;
}

Result<std::vector<float>> dequantize_group128(const PackedMatrix& matrix) {
  const auto expected_codes = matrix.rows * packed_row_bytes(matrix.format, matrix.columns);
  const auto expected_scales = matrix.rows * (matrix.columns / kGroupSize);
  if (matrix.rows == 0 || matrix.columns == 0 || matrix.columns % kGroupSize != 0 ||
      matrix.codes.size() != expected_codes || matrix.scales.size() != expected_scales) {
    return Status{ErrorCode::invalid_argument, "packed matrix metadata or storage is invalid"};
  }
  std::vector<float> output(matrix.rows * matrix.columns);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t column = 0; column < matrix.columns; ++column) {
      output[row * matrix.columns + column] = dequantized_weight(matrix, row, column);
    }
  }
  return output;
}

Result<Gfx12TiledMatrix> tile_group128_for_gfx12(const PackedMatrix& matrix) {
  const auto row_bytes = packed_row_bytes(matrix.format, matrix.columns);
  const auto groups = matrix.columns / kGroupSize;
  if (matrix.rows == 0 || matrix.columns == 0 || matrix.rows % 16 != 0 ||
      matrix.columns % kGroupSize != 0 ||
      (matrix.format != Quantization::w2g128 &&
       matrix.format != Quantization::w3g128 &&
       matrix.format != Quantization::w4g128) ||
      matrix.codes.size() != matrix.rows * row_bytes ||
      matrix.scales.size() != matrix.rows * groups) {
    return Status{ErrorCode::invalid_argument,
                  "gfx12 tiling requires valid W2G128/W3G128/W4G128 storage, N divisible by 16, "
                  "and K divisible by 128"};
  }

  Gfx12TiledMatrix output;
  output.rows = matrix.rows;
  output.columns = matrix.columns;
  output.format = matrix.format;
  output.codes.resize(matrix.codes.size());
  output.scales.resize(matrix.scales.size());
  const auto fragment_bytes = matrix.format == Quantization::w2g128
                                  ? 8U
                                  : (matrix.format == Quantization::w3g128 ? 12U : 16U);
  std::size_t code_destination = 0;
  for (std::size_t tile = 0; tile < matrix.rows / 16; ++tile) {
    for (std::size_t group = 0; group < groups; ++group) {
      for (std::size_t k_tile = 0; k_tile < 4; ++k_tile) {
        for (std::size_t local_row = 0; local_row < 16; ++local_row) {
          const auto row = tile * 16 + local_row;
          if (matrix.format == Quantization::w3g128) {
            for (std::size_t item = 0; item < 32; ++item) {
              const auto column = group * kGroupSize + k_tile * 32U + item;
              const auto code = load_code(
                  std::span<const std::uint8_t>(matrix.codes).subspan(
                      row * row_bytes, row_bytes),
                  column, matrix.format);
              const auto bit = item * 3U;
              const auto byte = bit / 8U;
              const auto shift = static_cast<unsigned>(bit & 7U);
              output.codes[code_destination + byte] |=
                  static_cast<std::uint8_t>(code << shift);
              if (shift > 5U) {
                output.codes[code_destination + byte + 1U] |=
                    static_cast<std::uint8_t>(code >> (8U - shift));
              }
            }
          } else {
            const auto source = row * row_bytes +
                                group * fragment_bytes * 4 +
                                k_tile * fragment_bytes;
            std::copy_n(matrix.codes.begin() + static_cast<std::ptrdiff_t>(source),
                        fragment_bytes,
                        output.codes.begin() + static_cast<std::ptrdiff_t>(code_destination));
          }
          code_destination += fragment_bytes;
        }
      }
      for (std::size_t local_row = 0; local_row < 16; ++local_row) {
        output.scales[(tile * groups + group) * 16 + local_row] =
            matrix.scales[(tile * 16 + local_row) * groups + group];
      }
    }
  }
  return output;
}

}  // namespace gfxinfer
