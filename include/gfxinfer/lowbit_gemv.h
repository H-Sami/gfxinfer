// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include "gfxinfer/artifact.h"
#include "gfxinfer/status.h"

namespace gfxinfer {

[[nodiscard]] Status launch_lowbit_gemv(
    Quantization format,
    const std::uint8_t* packed_weights,
    const __half* scales,
    const __half* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream);

[[nodiscard]] Status launch_lowbit_row_gather(
    Quantization format,
    const std::uint8_t* packed_weights,
    const __half* scales,
    __half* output,
    std::size_t row,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream);

[[nodiscard]] Status launch_lowbit_row_gather_indirect(
    Quantization format,
    const std::uint8_t* packed_weights,
    const __half* scales,
    __half* output,
    const std::uint32_t* row,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream);

[[nodiscard]] Status launch_lowbit_row_gather_tiled(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_scales,
    __half* output,
    std::size_t row,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream);

[[nodiscard]] Status launch_lowbit_row_gather_tiled_indirect(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_scales,
    __half* output,
    const std::uint32_t* row,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream);

// Input and output are batch-major: input[B,K], output[B,N]. The matrix is
// fetched once per output row and reused across every batch column.
[[nodiscard]] Status launch_lowbit_gemv_batch(
    Quantization format,
    const std::uint8_t* packed_weights,
    const __half* scales,
    const __half* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream);

// Quantize a batch-major FP16 activation matrix to signed A4G128. The packed
// output is [B,K/2], and the FP16 scales are [B,K/128]. B may be 1..32.
[[nodiscard]] Status launch_quantize_activation_i4(
    const __half* input,
    std::uint8_t* packed_output,
    __half* scales,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream);

[[nodiscard]] Status launch_quantize_activation_i8(
    const __half* input,
    std::int8_t* quantized_output,
    __half* scales,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream);

// Native RDNA 4 16x16x32 INT4 wave-matrix path. Weight codes and scales must
// use tile_group128_for_gfx12() order; activations use the layout produced by
// launch_quantize_activation_i4(). Output is batch-major [B,N].
[[nodiscard]] Status launch_gfx12_i4_gemm(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_weight_scales,
    const std::uint8_t* packed_activations,
    const __half* activation_scales,
    float* output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch,
    bool sequential_w3,
    hipStream_t stream);

// Same RDNA 4 WMMA path with an FP16 epilogue. This is useful when the next
// operator consumes FP16 and avoids materializing an intermediate FP32 matrix.
[[nodiscard]] Status launch_gfx12_i4_gemm_half(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_weight_scales,
    const std::uint8_t* packed_activations,
    const __half* activation_scales,
    __half* output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch,
    bool sequential_w3,
    hipStream_t stream);

struct Gfx12I4FusedMatrix {
  const std::uint8_t* tiled_weights{nullptr};
  const __half* tiled_weight_scales{nullptr};
  float* output{nullptr};
  std::size_t rows{0};
};

// Executes same-input projections in one grid and reuses the already
// quantized activation. Matrices must share K and weight format.
[[nodiscard]] Status launch_gfx12_i4_gemm_fused(
    Quantization format,
    std::span<const Gfx12I4FusedMatrix> matrices,
    const std::uint8_t* packed_activations,
    const __half* activation_scales,
    std::size_t columns,
    unsigned batch,
    bool sequential_w3,
    hipStream_t stream);

[[nodiscard]] Status launch_gfx12_i8_gemm(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_weight_scales,
    const std::int8_t* quantized_activations,
    const __half* activation_scales,
    float* output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream);

// Native FP16 WMMA path with exact FP16 activations. Packed W2/W4 codes are
// expanded wave-locally and multiplied by their group scales before MMA.
[[nodiscard]] Status launch_gfx12_f16_gemm(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_weight_scales,
    const __half* activations,
    float* output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream);

}  // namespace gfxinfer
