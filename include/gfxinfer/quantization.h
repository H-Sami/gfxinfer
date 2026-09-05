// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "gfxinfer/artifact.h"
#include "gfxinfer/status.h"

namespace gfxinfer {

struct PackedMatrix {
  std::size_t rows{0};
  std::size_t columns{0};
  Quantization format{Quantization::none};
  std::vector<std::uint8_t> codes;
  std::vector<float> scales;
};

// Storage order consumed directly by the RDNA 4 wave-matrix kernels. Codes
// are ordered [row-tile-16, group-128, k-tile-32, row-in-tile, fragment-byte]
// and scales are [row-tile-16, group-128, row-in-tile]. This keeps every
// wave's weight loads contiguous without changing the packed bit count.
struct Gfx12TiledMatrix {
  std::size_t rows{0};
  std::size_t columns{0};
  Quantization format{Quantization::none};
  std::vector<std::uint8_t> codes;
  std::vector<float> scales;
};

[[nodiscard]] Result<PackedMatrix> quantize_group128(
    std::span<const float> weights,
    std::size_t rows,
    std::size_t columns,
    Quantization format);

[[nodiscard]] Result<std::vector<float>> dequantize_group128(
    const PackedMatrix& matrix);

[[nodiscard]] Result<Gfx12TiledMatrix> tile_group128_for_gfx12(
    const PackedMatrix& matrix);

[[nodiscard]] float dequantized_weight(
    const PackedMatrix& matrix,
    std::size_t row,
    std::size_t column);

[[nodiscard]] std::size_t packed_row_bytes(
    Quantization format,
    std::size_t columns);

}  // namespace gfxinfer
