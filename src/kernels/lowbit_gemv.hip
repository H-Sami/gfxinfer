// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/lowbit_gemv.h"

#include "gfxinfer/device.h"

#include <hip/hip_runtime.h>

#include <limits>
#include <string>

namespace gfxinfer {
namespace {

constexpr unsigned kWaveSize = 32;
constexpr std::size_t kGroupSize = 128;

__device__ __forceinline__ float wave_sum(float value) {
  value += __shfl_down(value, 16, kWaveSize);
  value += __shfl_down(value, 8, kWaveSize);
  value += __shfl_down(value, 4, kWaveSize);
  value += __shfl_down(value, 2, kWaveSize);
  value += __shfl_down(value, 1, kWaveSize);
  return value;
}

__global__ __launch_bounds__(32) void w2g128_gemv(
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ scales,
    const __half* __restrict__ input,
    float* __restrict__ output,
    std::size_t columns) {
  const unsigned lane = threadIdx.x;
  const std::size_t row = blockIdx.x;
  const std::size_t groups = columns / kGroupSize;
  const std::size_t row_bytes = columns / 4;
  const auto* row_weights = weights + row * row_bytes;
  float sum = 0.0F;
  std::size_t group = 0;
  for (; group + 4 <= groups; group += 4) {
    // Four adjacent groups form one coalesced 128-byte wave transaction. A
    // one-group loop only requests 32 bytes and leaves most of each cache line
    // unused on this memory system.
    const auto* words = reinterpret_cast<const std::uint32_t*>(row_weights + group * 32);
    const std::uint32_t packed_word = words[lane];
#pragma unroll
    for (unsigned byte_in_word = 0; byte_in_word < 4; ++byte_in_word) {
      const unsigned global_byte = lane * 4U + byte_in_word;
      const unsigned local_group = global_byte / 32U;
      const unsigned byte_in_group = global_byte & 31U;
      const std::uint8_t packed = static_cast<std::uint8_t>(
          packed_word >> (byte_in_word * 8U));
      const float scale = __half2float(scales[row * groups + group + local_group]);
      const std::size_t input_base =
          (group + local_group) * kGroupSize + byte_in_group * 4U;
#pragma unroll
      for (unsigned sub = 0; sub < 4; ++sub) {
        const auto code = static_cast<int>((packed >> (sub * 2U)) & 0x03U);
        const float weight = static_cast<float>(code * 2 - 3) * scale;
        sum = fmaf(weight, __half2float(input[input_base + sub]), sum);
      }
    }
  }
  for (; group < groups; ++group) {
    const float scale = __half2float(scales[row * groups + group]);
    const std::uint8_t packed = row_weights[group * 32 + lane];
    const std::size_t input_base = group * kGroupSize + lane * 4U;
#pragma unroll
    for (unsigned sub = 0; sub < 4; ++sub) {
      const auto code = static_cast<int>((packed >> (sub * 2U)) & 0x03U);
      sum = fmaf(static_cast<float>(code * 2 - 3) * scale,
                 __half2float(input[input_base + sub]), sum);
    }
  }
  sum = wave_sum(sum);
  if (lane == 0) {
    output[row] = sum;
  }
}

__global__ __launch_bounds__(256) void w3g128_gemv(
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ scales,
    const __half* __restrict__ input,
    float* __restrict__ output,
    std::size_t rows,
    std::size_t columns) {
  const unsigned lane = threadIdx.x & (kWaveSize - 1U);
  const unsigned wave = threadIdx.x / kWaveSize;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * 8U + wave;
  if (row >= rows) return;
  const std::size_t groups = columns / kGroupSize;
  const std::size_t row_bytes = columns * 3 / 8;
  const auto* row_weights = weights + row * row_bytes;
  float sum = 0.0F;
  for (std::size_t group = 0; group < groups; ++group) {
    const float scale = __half2float(scales[row * groups + group]);
    const auto* group_weights = row_weights + group * 48;
    unsigned plane0 = lane < 16 ? group_weights[lane] : 0U;
    unsigned plane1 = lane < 16 ? group_weights[16 + lane] : 0U;
    unsigned plane2 = lane < 16 ? group_weights[32 + lane] : 0U;
    const unsigned source_lane = lane >> 1U;
    plane0 = __shfl(plane0, source_lane, kWaveSize);
    plane1 = __shfl(plane1, source_lane, kWaveSize);
    plane2 = __shfl(plane2, source_lane, kWaveSize);
    const unsigned first_bit = (lane & 1U) * 4U;
#pragma unroll
    for (unsigned item = 0; item < 4; ++item) {
        const unsigned bit = first_bit + item;
        const unsigned code = ((plane0 >> bit) & 1U) |
                              (((plane1 >> bit) & 1U) << 1U) |
                              (((plane2 >> bit) & 1U) << 2U);
        const int quantized = static_cast<int>(code) - 4;
        const auto column = group * kGroupSize + source_lane * 8U + bit;
        sum = fmaf(static_cast<float>(quantized) * scale,
                   __half2float(input[column]), sum);
    }
  }
  sum = wave_sum(sum);
  if (lane == 0) {
    output[row] = sum;
  }
}

__global__ __launch_bounds__(32) void w4g128_gemv(
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ scales,
    const __half* __restrict__ input,
    float* __restrict__ output,
    std::size_t columns) {
  const unsigned lane = threadIdx.x;
  const std::size_t row = blockIdx.x;
  const std::size_t groups = columns / kGroupSize;
  const std::size_t row_bytes = columns / 2;
  const auto* row_weights = weights + row * row_bytes;
  float sum = 0.0F;
  for (std::size_t group = 0; group < groups; ++group) {
    const float scale = __half2float(scales[row * groups + group]);
#pragma unroll
    for (unsigned item = 0; item < 2; ++item) {
      const unsigned byte_in_group = lane + item * kWaveSize;
      const std::uint8_t packed = row_weights[group * 64 + byte_in_group];
      const std::size_t input_base = group * kGroupSize + byte_in_group * 2;
#pragma unroll
      for (unsigned sub = 0; sub < 2; ++sub) {
        const auto code = static_cast<int>((packed >> (sub * 4U)) & 0x0fU);
        const int quantized = code >= 8 ? code - 16 : code;
        sum = fmaf(static_cast<float>(quantized) * scale,
                   __half2float(input[input_base + sub]), sum);
      }
    }
  }
  sum = wave_sum(sum);
  if (lane == 0) {
    output[row] = sum;
  }
}

__global__ void lowbit_row_gather(
    Quantization format,
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    std::size_t row,
    const std::uint32_t* __restrict__ indirect_row,
    std::size_t rows,
    std::size_t columns) {
  const auto column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) return;
  if (indirect_row != nullptr) row = *indirect_row;
  if (row >= rows) {
    output[column] = __float2half(0.0F);
    return;
  }
  const std::size_t groups = columns / kGroupSize;
  const auto row_bytes = format == Quantization::w2g128 ? columns / 4 : columns / 2;
  const auto* row_weights = weights + row * row_bytes;
  int quantized = 0;
  if (format == Quantization::w2g128) {
    const auto packed = row_weights[column / 4];
    const auto code = static_cast<int>((packed >> ((column & 3U) * 2U)) & 0x03U);
    quantized = code * 2 - 3;
  } else {
    const auto packed = row_weights[column / 2];
    const auto code = static_cast<int>((packed >> ((column & 1U) * 4U)) & 0x0fU);
    quantized = code >= 8 ? code - 16 : code;
  }
  output[column] = __float2half(
      static_cast<float>(quantized) * __half2float(scales[row * groups + column / kGroupSize]));
}

__global__ void lowbit_tiled_row_gather(
    Quantization format,
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    std::size_t row,
    const std::uint32_t* __restrict__ indirect_row,
    std::size_t rows,
    std::size_t columns) {
  const auto column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) return;
  if (indirect_row != nullptr) row = *indirect_row;
  if (row >= rows) {
    output[column] = __float2half(0.0F);
    return;
  }
  const std::size_t groups = columns / kGroupSize;
  const std::size_t row_tile = row / 16U;
  const std::size_t local_row = row & 15U;
  const std::size_t group = column / kGroupSize;
  const std::size_t group_column = column & (kGroupSize - 1U);
  const std::size_t k_tile = group_column / 32U;
  const std::size_t local_column = group_column & 31U;
  const std::size_t fragment_bytes = format == Quantization::w2g128 ? 8U : 16U;
  const std::size_t fragment =
      ((((row_tile * groups + group) * 4U + k_tile) * 16U + local_row) * fragment_bytes);
  int quantized = 0;
  if (format == Quantization::w2g128) {
    const auto packed = weights[fragment + local_column / 4U];
    const auto code = static_cast<int>((packed >> ((local_column & 3U) * 2U)) & 0x03U);
    quantized = code * 2 - 3;
  } else {
    const auto packed = weights[fragment + local_column / 2U];
    const auto code = static_cast<int>((packed >> ((local_column & 1U) * 4U)) & 0x0fU);
    quantized = code >= 8 ? code - 16 : code;
  }
  output[column] = __float2half(
      static_cast<float>(quantized) *
      __half2float(scales[(row_tile * groups + group) * 16U + local_row]));
}

template <unsigned Batch, unsigned Bits>
__global__ __launch_bounds__(32) void lowbit_gemv_batch(
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ scales,
    const __half* __restrict__ input,
    float* __restrict__ output,
    std::size_t rows,
    std::size_t columns) {
  const unsigned lane = threadIdx.x;
  const std::size_t row = blockIdx.x;
  const std::size_t groups = columns / kGroupSize;
  const std::size_t row_bytes = Bits == 2 ? columns / 4 : columns / 2;
  const auto* row_weights = weights + row * row_bytes;
  float sums[Batch]{};
  for (std::size_t group = 0; group < groups; ++group) {
    const float scale = __half2float(scales[row * groups + group]);
    if constexpr (Bits == 2) {
      const std::uint8_t packed = row_weights[group * 32 + lane];
      const std::size_t input_base = group * kGroupSize + lane * 4U;
#pragma unroll
      for (unsigned sub = 0; sub < 4; ++sub) {
        const auto code = static_cast<int>((packed >> (sub * 2U)) & 0x03U);
        const float weight = static_cast<float>(code * 2 - 3) * scale;
#pragma unroll
        for (unsigned batch = 0; batch < Batch; ++batch) {
          sums[batch] = fmaf(weight,
                             __half2float(input[batch * columns + input_base + sub]),
                             sums[batch]);
        }
      }
    } else {
#pragma unroll
      for (unsigned item = 0; item < 2; ++item) {
        const unsigned byte_in_group = lane + item * kWaveSize;
        const std::uint8_t packed = row_weights[group * 64 + byte_in_group];
        const std::size_t input_base = group * kGroupSize + byte_in_group * 2U;
#pragma unroll
        for (unsigned sub = 0; sub < 2; ++sub) {
          const auto code = static_cast<int>((packed >> (sub * 4U)) & 0x0fU);
          const int quantized = code >= 8 ? code - 16 : code;
          const float weight = static_cast<float>(quantized) * scale;
#pragma unroll
          for (unsigned batch = 0; batch < Batch; ++batch) {
            sums[batch] = fmaf(weight,
                               __half2float(input[batch * columns + input_base + sub]),
                               sums[batch]);
          }
        }
      }
    }
  }
#pragma unroll
  for (unsigned batch = 0; batch < Batch; ++batch) {
    sums[batch] = wave_sum(sums[batch]);
    if (lane == 0) output[batch * rows + row] = sums[batch];
  }
}

using i32x2 = int __attribute__((ext_vector_type(2)));
using i32x8 = int __attribute__((ext_vector_type(8)));
using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ i32x2 load_i32x2(const std::uint8_t* source) {
  i32x2 result;
  result[0] = reinterpret_cast<const int*>(source)[0];
  result[1] = reinterpret_cast<const int*>(source)[1];
  return result;
}

__device__ __forceinline__ std::uint32_t expand_w2_half(std::uint32_t packed) {
  // Dilate eight adjacent 2-bit codes into eight 4-bit lanes, then map
  // {0,1,2,3} to the signed nibble codebook {-3,-1,1,3}. This is six packed
  // bit operations instead of sixteen scalar extract/insert sequences.
  packed &= 0x0000ffffU;
  packed = (packed | (packed << 8U)) & 0x00ff00ffU;
  packed = (packed | (packed << 4U)) & 0x0f0f0f0fU;
  packed = (packed | (packed << 2U)) & 0x33333333U;
  const std::uint32_t inverted_high = (~packed) & 0x22222222U;
  return 0x11111111U | ((packed & 0x11111111U) << 1U) |
         (inverted_high << 1U) | (inverted_high << 2U);
}

__device__ __forceinline__ i32x2 expand_w2_fragment(std::uint32_t packed) {
  i32x2 result;
  result[0] = static_cast<int>(expand_w2_half(packed));
  result[1] = static_cast<int>(expand_w2_half(packed >> 16U));
  return result;
}

__device__ __forceinline__ std::uint32_t expand_w3_sequential_half(
    std::uint32_t packed) {
  // Insert one zero bit after every 3-bit code, expanding eight adjacent W3
  // values into eight nibbles. Then map unsigned {0..7} to the signed INT4
  // representation {-4..3} expected by the WMMA instruction.
  packed = (packed & 0x00000fffU) | ((packed & 0x00fff000U) << 4U);
  packed = (packed & 0x003f003fU) | ((packed & 0x0fc00fc0U) << 2U);
  packed = (packed & 0x07070707U) | ((packed & 0x38383838U) << 1U);
  const std::uint32_t inverted = (~packed) & 0x44444444U;
  return (packed & 0x33333333U) | inverted | (inverted << 1U);
}

__device__ __forceinline__ i32x2 expand_w3_sequential_fragment(
    const std::uint8_t* source) {
  const std::uint32_t low =
      *reinterpret_cast<const std::uint32_t*>(source);
  const std::uint32_t high =
      *reinterpret_cast<const std::uint16_t*>(source + 4U);
  i32x2 result;
  result[0] = static_cast<int>(expand_w3_sequential_half(low));
  result[1] = static_cast<int>(expand_w3_sequential_half(
      (low >> 24U) | (high << 8U)));
  return result;
}

__device__ __forceinline__ i32x2 expand_w3_bitplane_fragment(
    const std::uint8_t* source) {
  auto dilate8 = [](std::uint32_t value) {
    value &= 0xffU;
    value = (value | (value << 12U)) & 0x000f000fU;
    value = (value | (value << 6U)) & 0x03030303U;
    value = (value | (value << 3U)) & 0x11111111U;
    return value;
  };
  const std::uint32_t low =
      *reinterpret_cast<const std::uint32_t*>(source);
  const std::uint32_t high =
      *reinterpret_cast<const std::uint16_t*>(source + 4U);
  i32x2 result;
#pragma unroll
  for (unsigned half = 0; half < 2; ++half) {
    const std::uint32_t bit0 = dilate8(low >> (half * 8U));
    const std::uint32_t bit1 = dilate8(low >> ((half + 2U) * 8U));
    const std::uint32_t bit2 = dilate8(high >> (half * 8U));
    const std::uint32_t inverted = (~bit2) & 0x11111111U;
    result[half] = static_cast<int>(
        bit0 | (bit1 << 1U) | (inverted << 2U) | (inverted << 3U));
  }
  return result;
}

__global__ __launch_bounds__(128) void quantize_activation_i4(
    const __half* __restrict__ input,
    std::uint8_t* __restrict__ packed_output,
    __half* __restrict__ scales,
    std::size_t columns,
    std::size_t groups) {
  __shared__ float reduction[kGroupSize];
  const unsigned lane = threadIdx.x;
  const std::size_t linear_group = blockIdx.x;
  const std::size_t batch = linear_group / groups;
  const std::size_t group = linear_group - batch * groups;
  const std::size_t source_base = batch * columns + group * kGroupSize;
  reduction[lane] = fabsf(__half2float(input[source_base + lane]));
  __syncthreads();
  for (unsigned stride = 64; stride != 0; stride >>= 1U) {
    if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
    __syncthreads();
  }
  if (lane == 0) {
    const float scale = reduction[0] == 0.0F ? 1.0F : reduction[0] / 7.0F;
    scales[linear_group] = __float2half(scale);
  }
  __syncthreads();
  if (lane < 64) {
    // Use the rounded FP16 scale in both quantization and dequantization so
    // the WMMA result has one exact, reproducible scale contract.
    const float scale = __half2float(scales[linear_group]);
    int low = __float2int_rn(__half2float(input[source_base + lane * 2U]) / scale);
    int high = __float2int_rn(__half2float(input[source_base + lane * 2U + 1U]) / scale);
    low = max(-7, min(7, low));
    high = max(-7, min(7, high));
    packed_output[batch * (columns / 2) + group * 64 + lane] =
        static_cast<std::uint8_t>((low & 0x0f) | ((high & 0x0f) << 4U));
  }
}

__global__ __launch_bounds__(128) void quantize_activation_i8(
    const __half* __restrict__ input,
    std::int8_t* __restrict__ quantized_output,
    __half* __restrict__ scales,
    std::size_t columns,
    std::size_t groups) {
  __shared__ float reduction[kGroupSize];
  const unsigned lane = threadIdx.x;
  const std::size_t linear_group = blockIdx.x;
  const std::size_t batch = linear_group / groups;
  const std::size_t group = linear_group - batch * groups;
  const std::size_t offset = batch * columns + group * kGroupSize + lane;
  reduction[lane] = fabsf(__half2float(input[offset]));
  __syncthreads();
  for (unsigned stride = 64; stride != 0; stride >>= 1U) {
    if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
    __syncthreads();
  }
  if (lane == 0) {
    const float scale = reduction[0] == 0.0F ? 1.0F : reduction[0] / 127.0F;
    scales[linear_group] = __float2half(scale);
  }
  __syncthreads();
  const float scale = __half2float(scales[linear_group]);
  int quantized = __float2int_rn(__half2float(input[offset]) / scale);
  quantized = max(-127, min(127, quantized));
  quantized_output[offset] = static_cast<std::int8_t>(quantized);
}

__device__ __forceinline__ i32x2 expand_w2_i8(std::uint16_t packed) {
  i32x2 result{};
  auto* unpacked = reinterpret_cast<std::int8_t*>(&result);
#pragma unroll
  for (unsigned item = 0; item < 8; ++item) {
    const unsigned code = (packed >> (item * 2U)) & 3U;
    unpacked[item] = static_cast<std::int8_t>(static_cast<int>(code * 2U) - 3);
  }
  return result;
}

__device__ __forceinline__ i32x2 expand_w4_i8(std::uint32_t packed) {
  i32x2 result{};
  auto* unpacked = reinterpret_cast<std::int8_t*>(&result);
#pragma unroll
  for (unsigned item = 0; item < 8; ++item) {
    const unsigned code = (packed >> (item * 4U)) & 0x0fU;
    unpacked[item] = static_cast<std::int8_t>(
        code >= 8U ? static_cast<int>(code) - 16 : static_cast<int>(code));
  }
  return result;
}

template <unsigned WeightBits, bool HalfOutput = false,
          bool SequentialW3 = true>
__global__ __launch_bounds__(256) void gfx12_i4_gemm(
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activations,
    const __half* __restrict__ activation_scales,
    void* __restrict__ output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch) {
  const unsigned row_tiles_per_block = batch <= 4 ? 8U : 4U;
  __shared__ i32x2 expanded_weights[256];
  const unsigned lane = threadIdx.x & 31U;
  const unsigned physical_wave = threadIdx.x >> 5U;
  const unsigned waves_per_tile = batch > 16 ? 2U : 1U;
  const unsigned tile_in_block = physical_wave / waves_per_tile;
  const unsigned candidate_wave = physical_wave - tile_in_block * waves_per_tile;
  const unsigned candidate = candidate_wave * 16U + (lane & 15U);
  const unsigned row_half = lane >> 4U;
  const std::size_t row_tile =
      static_cast<std::size_t>(blockIdx.x) * row_tiles_per_block + tile_in_block;
  if (row_tile >= rows / 16U) return;
  const std::size_t groups = columns / kGroupSize;
  constexpr std::size_t fragment_bytes =
      WeightBits == 2 ? 8U : (WeightBits == 3 ? 12U : 16U);
  constexpr std::size_t lane_bytes = fragment_bytes / 2U;
  float accumulators[8]{};

  for (std::size_t group = 0; group < groups; ++group) {
    i32x8 dot{};
#pragma unroll
    for (unsigned k_tile = 0; k_tile < 4; ++k_tile) {
      const std::size_t weight_offset =
          ((((row_tile * groups + group) * 4U + k_tile) * 16U + (lane & 15U)) *
           fragment_bytes) +
          row_half * lane_bytes;
      i32x2 a;
      if constexpr (WeightBits == 2) {
        const auto packed_w2 =
            *reinterpret_cast<const std::uint32_t*>(weights + weight_offset);
        expanded_weights[threadIdx.x] = expand_w2_fragment(packed_w2);
        __syncwarp();
        a = expanded_weights[threadIdx.x];
      } else if constexpr (WeightBits == 3) {
        expanded_weights[threadIdx.x] = SequentialW3
            ? expand_w3_sequential_fragment(weights + weight_offset)
            : expand_w3_bitplane_fragment(weights + weight_offset);
        __syncwarp();
        a = expanded_weights[threadIdx.x];
      } else {
        a = load_i32x2(weights + weight_offset);
      }

      i32x2 b{};
      if (candidate < batch) {
        const std::size_t activation_offset =
            static_cast<std::size_t>(candidate) * (columns / 2U) +
            group * 64U + k_tile * 16U + row_half * 8U;
        b = load_i32x2(activations + activation_offset);
      }
      dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
          true, a, true, b, dot, false);
    }
    if (candidate < batch) {
      const float activation_scale = __half2float(
          activation_scales[static_cast<std::size_t>(candidate) * groups + group]);
#pragma unroll
      for (unsigned item = 0; item < 8; ++item) {
        const unsigned local_row = row_half * 8U + item;
        const float weight_scale = __half2float(
            weight_scales[(row_tile * groups + group) * 16U + local_row]);
        accumulators[item] = fmaf(static_cast<float>(dot[item]),
                                  weight_scale * activation_scale,
                                  accumulators[item]);
      }
    }
  }

  if (candidate < batch) {
#pragma unroll
    for (unsigned item = 0; item < 8; ++item) {
      const std::size_t row = row_tile * 16U + row_half * 8U + item;
      const auto output_index = static_cast<std::size_t>(candidate) * rows + row;
      if constexpr (HalfOutput) {
        reinterpret_cast<__half*>(output)[output_index] =
            __float2half(accumulators[item]);
      } else {
        reinterpret_cast<float*>(output)[output_index] = accumulators[item];
      }
    }
  }
}

template <unsigned WeightBits, bool SequentialW3 = true>
__global__ __launch_bounds__(256) void gfx12_i4_gemm_fused(
    const std::uint8_t* __restrict__ weights0,
    const __half* __restrict__ scales0,
    float* __restrict__ output0,
    std::size_t rows0,
    const std::uint8_t* __restrict__ weights1,
    const __half* __restrict__ scales1,
    float* __restrict__ output1,
    std::size_t rows1,
    const std::uint8_t* __restrict__ weights2,
    const __half* __restrict__ scales2,
    float* __restrict__ output2,
    std::size_t rows2,
    const std::uint8_t* __restrict__ weights3,
    const __half* __restrict__ scales3,
    float* __restrict__ output3,
    std::size_t rows3,
    const std::uint8_t* __restrict__ activations,
    const __half* __restrict__ activation_scales,
    std::size_t columns,
    unsigned batch) {
  const unsigned row_tiles_per_block = batch <= 4 ? 8U : 4U;
  __shared__ i32x2 expanded_weights[256];
  const unsigned lane = threadIdx.x & 31U;
  const unsigned physical_wave = threadIdx.x >> 5U;
  const unsigned waves_per_tile = batch > 16 ? 2U : 1U;
  const unsigned tile_in_block = physical_wave / waves_per_tile;
  const unsigned candidate_wave = physical_wave - tile_in_block * waves_per_tile;
  const unsigned candidate = candidate_wave * 16U + (lane & 15U);
  const unsigned row_half = lane >> 4U;
  std::size_t row_tile =
      static_cast<std::size_t>(blockIdx.x) * row_tiles_per_block + tile_in_block;
  const auto tiles0 = rows0 / 16U;
  const auto tiles1 = rows1 / 16U;
  const auto tiles2 = rows2 / 16U;
  const auto tiles3 = rows3 / 16U;
  if (row_tile >= tiles0 + tiles1 + tiles2 + tiles3) return;

  const std::uint8_t* weights = weights0;
  const __half* weight_scales = scales0;
  float* output = output0;
  std::size_t rows = rows0;
  if (row_tile >= tiles0) {
    row_tile -= tiles0;
    weights = weights1;
    weight_scales = scales1;
    output = output1;
    rows = rows1;
    if (row_tile >= tiles1) {
      row_tile -= tiles1;
      weights = weights2;
      weight_scales = scales2;
      output = output2;
      rows = rows2;
      if (row_tile >= tiles2) {
        row_tile -= tiles2;
        weights = weights3;
        weight_scales = scales3;
        output = output3;
        rows = rows3;
      }
    }
  }

  const std::size_t groups = columns / kGroupSize;
  constexpr std::size_t fragment_bytes =
      WeightBits == 2 ? 8U : (WeightBits == 3 ? 12U : 16U);
  constexpr std::size_t lane_bytes = fragment_bytes / 2U;
  float accumulators[8]{};
  for (std::size_t group = 0; group < groups; ++group) {
    i32x8 dot{};
#pragma unroll
    for (unsigned k_tile = 0; k_tile < 4; ++k_tile) {
      const std::size_t weight_offset =
          ((((row_tile * groups + group) * 4U + k_tile) * 16U + (lane & 15U)) *
           fragment_bytes) +
          row_half * lane_bytes;
      i32x2 a;
      if constexpr (WeightBits == 2) {
        const auto packed =
            *reinterpret_cast<const std::uint32_t*>(weights + weight_offset);
        expanded_weights[threadIdx.x] = expand_w2_fragment(packed);
        __syncwarp();
        a = expanded_weights[threadIdx.x];
      } else if constexpr (WeightBits == 3) {
        expanded_weights[threadIdx.x] = SequentialW3
            ? expand_w3_sequential_fragment(weights + weight_offset)
            : expand_w3_bitplane_fragment(weights + weight_offset);
        __syncwarp();
        a = expanded_weights[threadIdx.x];
      } else {
        a = load_i32x2(weights + weight_offset);
      }
      i32x2 b{};
      if (candidate < batch) {
        const std::size_t activation_offset =
            static_cast<std::size_t>(candidate) * (columns / 2U) +
            group * 64U + k_tile * 16U + row_half * 8U;
        b = load_i32x2(activations + activation_offset);
      }
      dot = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
          true, a, true, b, dot, false);
    }
    if (candidate < batch) {
      const float activation_scale = __half2float(
          activation_scales[static_cast<std::size_t>(candidate) * groups + group]);
#pragma unroll
      for (unsigned item = 0; item < 8; ++item) {
        const unsigned local_row = row_half * 8U + item;
        const float weight_scale = __half2float(
            weight_scales[(row_tile * groups + group) * 16U + local_row]);
        accumulators[item] = fmaf(static_cast<float>(dot[item]),
                                  weight_scale * activation_scale,
                                  accumulators[item]);
      }
    }
  }
  if (candidate < batch) {
#pragma unroll
    for (unsigned item = 0; item < 8; ++item) {
      const std::size_t row = row_tile * 16U + row_half * 8U + item;
      output[static_cast<std::size_t>(candidate) * rows + row] = accumulators[item];
    }
  }
}

template <unsigned WeightBits>
__global__ __launch_bounds__(64) void gfx12_i8_gemm(
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ weight_scales,
    const std::int8_t* __restrict__ activations,
    const __half* __restrict__ activation_scales,
    float* __restrict__ output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch) {
  __shared__ i32x2 expanded_weights[64];
  const unsigned lane = threadIdx.x & 31U;
  const unsigned wave = threadIdx.x >> 5U;
  const unsigned candidate = wave * 16U + (lane & 15U);
  const unsigned row_half = lane >> 4U;
  const std::size_t row_tile = blockIdx.x;
  const std::size_t groups = columns / kGroupSize;
  constexpr std::size_t fragment_bytes = WeightBits == 2 ? 8U : 16U;
  constexpr std::size_t source_bytes = WeightBits == 2 ? 2U : 4U;
  float accumulators[8]{};

  for (std::size_t group = 0; group < groups; ++group) {
    i32x8 dot{};
#pragma unroll
    for (unsigned k_tile = 0; k_tile < 4; ++k_tile) {
#pragma unroll
      for (unsigned k_half = 0; k_half < 2; ++k_half) {
        const std::size_t fragment =
            ((((row_tile * groups + group) * 4U + k_tile) * 16U + (lane & 15U)) *
             fragment_bytes);
        const std::size_t weight_offset =
            fragment + k_half * (fragment_bytes / 2U) + row_half * source_bytes;
        if constexpr (WeightBits == 2) {
          expanded_weights[threadIdx.x] = expand_w2_i8(
              *reinterpret_cast<const std::uint16_t*>(weights + weight_offset));
        } else {
          expanded_weights[threadIdx.x] = expand_w4_i8(
              *reinterpret_cast<const std::uint32_t*>(weights + weight_offset));
        }
        __syncthreads();
        const i32x2 a = expanded_weights[threadIdx.x];
        i32x2 b{};
        if (candidate < batch) {
          const std::size_t activation_offset =
              static_cast<std::size_t>(candidate) * columns + group * kGroupSize +
              k_tile * 32U + k_half * 16U + row_half * 8U;
          b = load_i32x2(reinterpret_cast<const std::uint8_t*>(activations + activation_offset));
        }
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, a, true, b, dot, false);
      }
    }
    if (candidate < batch) {
      const float activation_scale = __half2float(
          activation_scales[static_cast<std::size_t>(candidate) * groups + group]);
#pragma unroll
      for (unsigned item = 0; item < 8; ++item) {
        const unsigned local_row = row_half * 8U + item;
        const float weight_scale = __half2float(
            weight_scales[(row_tile * groups + group) * 16U + local_row]);
        accumulators[item] = fmaf(static_cast<float>(dot[item]),
                                  weight_scale * activation_scale,
                                  accumulators[item]);
      }
    }
  }
  if (candidate < batch) {
#pragma unroll
    for (unsigned item = 0; item < 8; ++item) {
      const std::size_t row = row_tile * 16U + row_half * 8U + item;
      output[static_cast<std::size_t>(candidate) * rows + row] = accumulators[item];
    }
  }
}

template <unsigned WeightBits>
__global__ __launch_bounds__(64) void gfx12_f16_gemm(
    const std::uint8_t* __restrict__ weights,
    const __half* __restrict__ weight_scales,
    const __half* __restrict__ activations,
    float* __restrict__ output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch) {
  __shared__ f16x8 expanded_weights[64];
  const unsigned lane = threadIdx.x & 31U;
  const unsigned wave = threadIdx.x >> 5U;
  const unsigned candidate = wave * 16U + (lane & 15U);
  const unsigned row_half = lane >> 4U;
  const std::size_t row_tile = blockIdx.x;
  const std::size_t groups = columns / kGroupSize;
  constexpr std::size_t fragment_bytes = WeightBits == 2 ? 8U : 16U;
  constexpr std::size_t source_bytes = WeightBits == 2 ? 2U : 4U;
  f32x8 accumulators{};

  for (std::size_t group = 0; group < groups; ++group) {
    const float weight_scale = __half2float(
        weight_scales[(row_tile * groups + group) * 16U + (lane & 15U)]);
#pragma unroll
    for (unsigned k_tile = 0; k_tile < 4; ++k_tile) {
#pragma unroll
      for (unsigned k_half = 0; k_half < 2; ++k_half) {
        const std::size_t fragment =
            ((((row_tile * groups + group) * 4U + k_tile) * 16U + (lane & 15U)) *
             fragment_bytes);
        const std::size_t weight_offset =
            fragment + k_half * (fragment_bytes / 2U) + row_half * source_bytes;
        f16x8 a;
        if constexpr (WeightBits == 2) {
          const auto packed =
              *reinterpret_cast<const std::uint16_t*>(weights + weight_offset);
#pragma unroll
          for (unsigned item = 0; item < 8; ++item) {
            const unsigned code = (packed >> (item * 2U)) & 3U;
            const int value = static_cast<int>(code * 2U) - 3;
            a[item] = static_cast<_Float16>(
                static_cast<float>(value) * weight_scale);
          }
        } else {
          const auto packed =
              *reinterpret_cast<const std::uint32_t*>(weights + weight_offset);
#pragma unroll
          for (unsigned item = 0; item < 8; ++item) {
            const unsigned code = (packed >> (item * 4U)) & 0x0fU;
            const int value = code >= 8 ? static_cast<int>(code) - 16
                                        : static_cast<int>(code);
            a[item] = static_cast<_Float16>(
                static_cast<float>(value) * weight_scale);
          }
        }
        expanded_weights[threadIdx.x] = a;
        __syncthreads();
        a = expanded_weights[threadIdx.x];
        f16x8 b{};
        if (candidate < batch) {
          const std::size_t activation_offset =
              static_cast<std::size_t>(candidate) * columns + group * kGroupSize +
              k_tile * 32U + k_half * 16U + row_half * 8U;
          b = *reinterpret_cast<const f16x8*>(activations + activation_offset);
        }
        accumulators = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            a, b, accumulators);
      }
    }
  }
  if (candidate < batch) {
#pragma unroll
    for (unsigned item = 0; item < 8; ++item) {
      const std::size_t row = row_tile * 16U + row_half * 8U + item;
      output[static_cast<std::size_t>(candidate) * rows + row] = accumulators[item];
    }
  }
}

}  // namespace

Status launch_lowbit_gemv(
    Quantization format,
    const std::uint8_t* packed_weights,
    const __half* scales,
    const __half* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream) {
  if (packed_weights == nullptr || scales == nullptr || input == nullptr || output == nullptr ||
      rows == 0 || columns == 0 || columns % kGroupSize != 0) {
    return {ErrorCode::invalid_argument,
            "low-bit GEMV requires non-null buffers, nonzero dimensions, and K divisible by 128"};
  }
  if (rows > std::numeric_limits<unsigned>::max()) {
    return {ErrorCode::invalid_argument, "GEMV row count exceeds the HIP grid limit"};
  }

  switch (format) {
    case Quantization::w2g128: {
      const dim3 grid(static_cast<unsigned>(rows));
      const dim3 block(kWaveSize);
      hipLaunchKernelGGL(w2g128_gemv, grid, block, 0, stream,
                         packed_weights, scales, input, output, columns);
      break;
    }
    case Quantization::w3g128: {
      const dim3 grid(static_cast<unsigned>((rows + 7U) / 8U));
      const dim3 block(256);
      hipLaunchKernelGGL(w3g128_gemv, grid, block, 0, stream,
                         packed_weights, scales, input, output, rows, columns);
      break;
    }
    case Quantization::w4g128: {
      const dim3 grid(static_cast<unsigned>(rows));
      const dim3 block(kWaveSize);
      hipLaunchKernelGGL(w4g128_gemv, grid, block, 0, stream,
                         packed_weights, scales, input, output, columns);
      break;
    }
    default:
      return {ErrorCode::unsupported, "requested low-bit GEMV format is not implemented"};
  }
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "low-bit GEMV launch failed");
  }
  return Status::ok_status();
}

Status launch_lowbit_row_gather(
    Quantization format,
    const std::uint8_t* packed_weights,
    const __half* scales,
    __half* output,
    std::size_t row,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream) {
  if (packed_weights == nullptr || scales == nullptr || output == nullptr ||
      row >= rows || columns == 0 || columns % kGroupSize != 0) {
    return {ErrorCode::invalid_argument, "invalid low-bit row gather arguments"};
  }
  if (format != Quantization::w2g128 && format != Quantization::w4g128) {
    return {ErrorCode::unsupported, "row gather supports W2G128 and W4G128"};
  }
  constexpr unsigned threads = 256;
  const dim3 grid(static_cast<unsigned>((columns + threads - 1U) / threads));
  hipLaunchKernelGGL(lowbit_row_gather, grid, dim3(threads), 0, stream,
                     format, packed_weights, scales, output, row, nullptr, rows, columns);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "low-bit row gather launch failed");
  }
  return Status::ok_status();
}

Status launch_lowbit_row_gather_indirect(
    Quantization format,
    const std::uint8_t* packed_weights,
    const __half* scales,
    __half* output,
    const std::uint32_t* row,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream) {
  if (packed_weights == nullptr || scales == nullptr || output == nullptr || row == nullptr ||
      rows == 0 || columns == 0 || columns % kGroupSize != 0) {
    return {ErrorCode::invalid_argument, "invalid indirect low-bit row gather arguments"};
  }
  if (format != Quantization::w2g128 && format != Quantization::w4g128) {
    return {ErrorCode::unsupported, "indirect row gather supports W2G128 and W4G128"};
  }
  constexpr unsigned threads = 256;
  const dim3 grid(static_cast<unsigned>((columns + threads - 1U) / threads));
  hipLaunchKernelGGL(lowbit_row_gather, grid, dim3(threads), 0, stream,
                     format, packed_weights, scales, output, 0U, row, rows, columns);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "indirect low-bit row gather launch failed");
  }
  return Status::ok_status();
}

Status launch_lowbit_row_gather_tiled(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_scales,
    __half* output,
    std::size_t row,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream) {
  if (tiled_weights == nullptr || tiled_scales == nullptr || output == nullptr ||
      row >= rows || rows % 16 != 0 || columns == 0 || columns % kGroupSize != 0) {
    return {ErrorCode::invalid_argument, "invalid tiled low-bit row gather arguments"};
  }
  if (format != Quantization::w2g128 && format != Quantization::w4g128) {
    return {ErrorCode::unsupported, "tiled row gather supports W2G128 and W4G128"};
  }
  constexpr unsigned threads = 256;
  const dim3 grid(static_cast<unsigned>((columns + threads - 1U) / threads));
  hipLaunchKernelGGL(lowbit_tiled_row_gather, grid, dim3(threads), 0, stream,
                     format, tiled_weights, tiled_scales, output, row, nullptr, rows, columns);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "tiled low-bit row gather launch failed");
  }
  return Status::ok_status();
}

Status launch_lowbit_row_gather_tiled_indirect(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_scales,
    __half* output,
    const std::uint32_t* row,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream) {
  if (tiled_weights == nullptr || tiled_scales == nullptr || output == nullptr || row == nullptr ||
      rows == 0 || rows % 16 != 0 || columns == 0 || columns % kGroupSize != 0) {
    return {ErrorCode::invalid_argument,
            "invalid indirect tiled low-bit row gather arguments"};
  }
  if (format != Quantization::w2g128 && format != Quantization::w4g128) {
    return {ErrorCode::unsupported,
            "indirect tiled row gather supports W2G128 and W4G128"};
  }
  constexpr unsigned threads = 256;
  const dim3 grid(static_cast<unsigned>((columns + threads - 1U) / threads));
  hipLaunchKernelGGL(lowbit_tiled_row_gather, grid, dim3(threads), 0, stream,
                     format, tiled_weights, tiled_scales, output, 0U, row, rows, columns);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "indirect tiled low-bit row gather launch failed");
  }
  return Status::ok_status();
}

namespace {

template <unsigned Batch, unsigned Bits>
void launch_batch_kernel(
    const std::uint8_t* packed_weights,
    const __half* scales,
    const __half* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream) {
  hipLaunchKernelGGL((lowbit_gemv_batch<Batch, Bits>),
                     dim3(static_cast<unsigned>(rows)), dim3(32), 0, stream,
                     packed_weights, scales, input, output, rows, columns);
}

template <unsigned Bits>
Status dispatch_batch(
    unsigned batch,
    const std::uint8_t* packed_weights,
    const __half* scales,
    const __half* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    hipStream_t stream) {
  switch (batch) {
    case 1: launch_batch_kernel<1, Bits>(packed_weights, scales, input, output, rows, columns, stream); break;
    case 2: launch_batch_kernel<2, Bits>(packed_weights, scales, input, output, rows, columns, stream); break;
    case 3: launch_batch_kernel<3, Bits>(packed_weights, scales, input, output, rows, columns, stream); break;
    case 4: launch_batch_kernel<4, Bits>(packed_weights, scales, input, output, rows, columns, stream); break;
    case 5: launch_batch_kernel<5, Bits>(packed_weights, scales, input, output, rows, columns, stream); break;
    case 6: launch_batch_kernel<6, Bits>(packed_weights, scales, input, output, rows, columns, stream); break;
    case 7: launch_batch_kernel<7, Bits>(packed_weights, scales, input, output, rows, columns, stream); break;
    case 8: launch_batch_kernel<8, Bits>(packed_weights, scales, input, output, rows, columns, stream); break;
    default: return {ErrorCode::invalid_argument, "low-bit batch must be in [1, 8]"};
  }
  return Status::ok_status();
}

}  // namespace

Status launch_lowbit_gemv_batch(
    Quantization format,
    const std::uint8_t* packed_weights,
    const __half* scales,
    const __half* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream) {
  if (packed_weights == nullptr || scales == nullptr || input == nullptr || output == nullptr ||
      rows == 0 || columns == 0 || columns % kGroupSize != 0 ||
      rows > std::numeric_limits<unsigned>::max()) {
    return {ErrorCode::invalid_argument, "invalid low-bit batched GEMV arguments"};
  }
  Status status;
  if (format == Quantization::w2g128) {
    status = dispatch_batch<2>(batch, packed_weights, scales, input, output,
                               rows, columns, stream);
  } else if (format == Quantization::w4g128) {
    status = dispatch_batch<4>(batch, packed_weights, scales, input, output,
                               rows, columns, stream);
  } else {
    return {ErrorCode::unsupported, "batched GEMV supports W2G128 and W4G128"};
  }
  if (!status.ok()) return status;
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "low-bit batched GEMV launch failed");
  }
  return Status::ok_status();
}

Status launch_quantize_activation_i4(
    const __half* input,
    std::uint8_t* packed_output,
    __half* scales,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream) {
  if (input == nullptr || packed_output == nullptr || scales == nullptr ||
      columns == 0 || columns % kGroupSize != 0 || batch == 0 || batch > 32) {
    return {ErrorCode::invalid_argument,
            "A4 quantization requires non-null buffers, batch in [1,32], and K divisible by 128"};
  }
  const auto groups = columns / kGroupSize;
  hipLaunchKernelGGL(quantize_activation_i4,
                     dim3(static_cast<unsigned>(groups * batch)), dim3(128), 0, stream,
                     input, packed_output, scales, columns, groups);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "A4 activation quantization launch failed");
  }
  return Status::ok_status();
}

Status launch_quantize_activation_i8(
    const __half* input,
    std::int8_t* quantized_output,
    __half* scales,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream) {
  if (input == nullptr || quantized_output == nullptr || scales == nullptr ||
      columns == 0 || columns % kGroupSize != 0 || batch == 0 || batch > 32) {
    return {ErrorCode::invalid_argument,
            "A8 quantization requires non-null buffers, batch in [1,32], and K divisible by 128"};
  }
  const auto groups = columns / kGroupSize;
  hipLaunchKernelGGL(quantize_activation_i8,
                     dim3(static_cast<unsigned>(groups * batch)), dim3(128), 0, stream,
                     input, quantized_output, scales, columns, groups);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "A8 activation quantization launch failed");
  }
  return Status::ok_status();
}

Status launch_gfx12_i4_gemm(
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
    hipStream_t stream) {
  if (tiled_weights == nullptr || tiled_weight_scales == nullptr ||
      packed_activations == nullptr || activation_scales == nullptr || output == nullptr ||
      rows == 0 || rows % 16 != 0 || columns == 0 || columns % kGroupSize != 0 ||
      batch == 0 || batch > 32 || rows / 16 > std::numeric_limits<unsigned>::max()) {
    return {ErrorCode::invalid_argument,
            "gfx12 INT4 GEMM requires tiled buffers, N divisible by 16, K divisible by 128, "
            "and batch in [1,32]"};
  }
  const unsigned row_tiles_per_block = batch <= 4 ? 8U : 4U;
  const auto row_tiles = static_cast<unsigned>(rows / 16);
  const dim3 grid((row_tiles + row_tiles_per_block - 1U) / row_tiles_per_block);
  const dim3 threads((batch <= 4 || batch > 16) ? 256U : 128U);
  if (format == Quantization::w2g128) {
    hipLaunchKernelGGL((gfx12_i4_gemm<2, false>), grid, threads, 0, stream,
                       tiled_weights, tiled_weight_scales, packed_activations,
                       activation_scales, output, rows, columns, batch);
  } else if (format == Quantization::w3g128) {
    if (sequential_w3) {
      hipLaunchKernelGGL((gfx12_i4_gemm<3, false, true>), grid, threads, 0,
                         stream, tiled_weights, tiled_weight_scales,
                         packed_activations, activation_scales, output, rows,
                         columns, batch);
    } else {
      hipLaunchKernelGGL((gfx12_i4_gemm<3, false, false>), grid, threads, 0,
                         stream, tiled_weights, tiled_weight_scales,
                         packed_activations, activation_scales, output, rows,
                         columns, batch);
    }
  } else if (format == Quantization::w4g128) {
    hipLaunchKernelGGL((gfx12_i4_gemm<4, false>), grid, threads, 0, stream,
                       tiled_weights, tiled_weight_scales, packed_activations,
                       activation_scales, output, rows, columns, batch);
  } else {
    return {ErrorCode::unsupported,
            "gfx12 INT4 GEMM supports W2G128, W3G128, and W4G128"};
  }
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "gfx12 INT4 GEMM launch failed");
  }
  return Status::ok_status();
}

Status launch_gfx12_i4_gemm_half(
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
    hipStream_t stream) {
  if (tiled_weights == nullptr || tiled_weight_scales == nullptr ||
      packed_activations == nullptr || activation_scales == nullptr ||
      output == nullptr || rows == 0 || rows % 16 != 0 || columns == 0 ||
      columns % kGroupSize != 0 || batch == 0 || batch > 32 ||
      rows / 16 > std::numeric_limits<unsigned>::max()) {
    return {ErrorCode::invalid_argument,
            "gfx12 INT4 GEMM FP16 epilogue requires tiled buffers, N divisible "
            "by 16, K divisible by 128, and batch in [1,32]"};
  }
  const unsigned row_tiles_per_block = batch <= 4 ? 8U : 4U;
  const auto row_tiles = static_cast<unsigned>(rows / 16);
  const dim3 grid((row_tiles + row_tiles_per_block - 1U) /
                  row_tiles_per_block);
  const dim3 threads((batch <= 4 || batch > 16) ? 256U : 128U);
  if (format == Quantization::w2g128) {
    hipLaunchKernelGGL((gfx12_i4_gemm<2, true>), grid, threads, 0, stream,
                       tiled_weights, tiled_weight_scales, packed_activations,
                       activation_scales, output, rows, columns, batch);
  } else if (format == Quantization::w3g128) {
    if (sequential_w3) {
      hipLaunchKernelGGL((gfx12_i4_gemm<3, true, true>), grid, threads, 0,
                         stream, tiled_weights, tiled_weight_scales,
                         packed_activations, activation_scales, output, rows,
                         columns, batch);
    } else {
      hipLaunchKernelGGL((gfx12_i4_gemm<3, true, false>), grid, threads, 0,
                         stream, tiled_weights, tiled_weight_scales,
                         packed_activations, activation_scales, output, rows,
                         columns, batch);
    }
  } else if (format == Quantization::w4g128) {
    hipLaunchKernelGGL((gfx12_i4_gemm<4, true>), grid, threads, 0, stream,
                       tiled_weights, tiled_weight_scales, packed_activations,
                       activation_scales, output, rows, columns, batch);
  } else {
    return {ErrorCode::unsupported,
            "gfx12 INT4 GEMM FP16 epilogue supports W2G128, W3G128, and W4G128"};
  }
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "gfx12 INT4 GEMM FP16 epilogue launch failed");
  }
  return Status::ok_status();
}

Status launch_gfx12_i4_gemm_fused(
    Quantization format,
    std::span<const Gfx12I4FusedMatrix> matrices,
    const std::uint8_t* packed_activations,
    const __half* activation_scales,
    std::size_t columns,
    unsigned batch,
    bool sequential_w3,
    hipStream_t stream) {
  if (matrices.size() < 2 || matrices.size() > 4 ||
      packed_activations == nullptr || activation_scales == nullptr ||
      columns == 0 || columns % kGroupSize != 0 || batch == 0 || batch > 32) {
    return {ErrorCode::invalid_argument,
            "gfx12 fused INT4 GEMM requires 2..4 matrices and batch in [1,32]"};
  }
  Gfx12I4FusedMatrix items[4]{};
  std::size_t total_row_tiles = 0;
  for (std::size_t index = 0; index < matrices.size(); ++index) {
    const auto& matrix = matrices[index];
    if (matrix.tiled_weights == nullptr || matrix.tiled_weight_scales == nullptr ||
        matrix.output == nullptr || matrix.rows == 0 || matrix.rows % 16 != 0) {
      return {ErrorCode::invalid_argument,
              "gfx12 fused INT4 GEMM matrix binding is invalid"};
    }
    items[index] = matrix;
    total_row_tiles += matrix.rows / 16U;
  }
  const unsigned row_tiles_per_block = batch <= 4 ? 8U : 4U;
  if (total_row_tiles >
      static_cast<std::size_t>(std::numeric_limits<unsigned>::max()) *
          row_tiles_per_block) {
    return {ErrorCode::invalid_argument, "gfx12 fused INT4 GEMM grid is too large"};
  }
  const dim3 grid(static_cast<unsigned>(
      (total_row_tiles + row_tiles_per_block - 1U) / row_tiles_per_block));
  const dim3 threads((batch <= 4 || batch > 16) ? 256U : 128U);
#define GFXINFER_FUSED_I4_ARGS                                                   \
  items[0].tiled_weights, items[0].tiled_weight_scales, items[0].output,        \
      items[0].rows, items[1].tiled_weights, items[1].tiled_weight_scales,      \
      items[1].output, items[1].rows, items[2].tiled_weights,                    \
      items[2].tiled_weight_scales, items[2].output, items[2].rows,              \
      items[3].tiled_weights, items[3].tiled_weight_scales, items[3].output,     \
      items[3].rows, packed_activations, activation_scales, columns, batch
  if (format == Quantization::w2g128) {
    hipLaunchKernelGGL((gfx12_i4_gemm_fused<2>), grid, threads, 0, stream,
                       GFXINFER_FUSED_I4_ARGS);
  } else if (format == Quantization::w3g128) {
    if (sequential_w3) {
      hipLaunchKernelGGL((gfx12_i4_gemm_fused<3, true>), grid, threads, 0,
                         stream, GFXINFER_FUSED_I4_ARGS);
    } else {
      hipLaunchKernelGGL((gfx12_i4_gemm_fused<3, false>), grid, threads, 0,
                         stream, GFXINFER_FUSED_I4_ARGS);
    }
  } else if (format == Quantization::w4g128) {
    hipLaunchKernelGGL((gfx12_i4_gemm_fused<4>), grid, threads, 0, stream,
                       GFXINFER_FUSED_I4_ARGS);
  } else {
#undef GFXINFER_FUSED_I4_ARGS
    return {ErrorCode::unsupported,
            "gfx12 fused INT4 GEMM supports W2G128, W3G128, and W4G128"};
  }
#undef GFXINFER_FUSED_I4_ARGS
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "gfx12 fused INT4 GEMM launch failed");
  }
  return Status::ok_status();
}

Status launch_gfx12_i8_gemm(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_weight_scales,
    const std::int8_t* quantized_activations,
    const __half* activation_scales,
    float* output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream) {
  if (tiled_weights == nullptr || tiled_weight_scales == nullptr ||
      quantized_activations == nullptr || activation_scales == nullptr || output == nullptr ||
      rows == 0 || rows % 16 != 0 || columns == 0 || columns % kGroupSize != 0 ||
      batch == 0 || batch > 32 || rows / 16 > std::numeric_limits<unsigned>::max()) {
    return {ErrorCode::invalid_argument,
            "gfx12 INT8 GEMM requires tiled buffers, N divisible by 16, K divisible by 128, "
            "and batch in [1,32]"};
  }
  const dim3 grid(static_cast<unsigned>(rows / 16));
  const dim3 threads(batch > 16 ? 64U : 32U);
  if (format == Quantization::w2g128) {
    hipLaunchKernelGGL((gfx12_i8_gemm<2>), grid, threads, 0, stream,
                       tiled_weights, tiled_weight_scales, quantized_activations,
                       activation_scales, output, rows, columns, batch);
  } else if (format == Quantization::w4g128) {
    hipLaunchKernelGGL((gfx12_i8_gemm<4>), grid, threads, 0, stream,
                       tiled_weights, tiled_weight_scales, quantized_activations,
                       activation_scales, output, rows, columns, batch);
  } else {
    return {ErrorCode::unsupported, "gfx12 INT8 GEMM supports W2G128 and W4G128"};
  }
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "gfx12 INT8 GEMM launch failed");
  }
  return Status::ok_status();
}

Status launch_gfx12_f16_gemm(
    Quantization format,
    const std::uint8_t* tiled_weights,
    const __half* tiled_weight_scales,
    const __half* activations,
    float* output,
    std::size_t rows,
    std::size_t columns,
    unsigned batch,
    hipStream_t stream) {
  if (tiled_weights == nullptr || tiled_weight_scales == nullptr ||
      activations == nullptr || output == nullptr || rows == 0 || rows % 16 != 0 ||
      columns == 0 || columns % kGroupSize != 0 || batch == 0 || batch > 32 ||
      rows / 16 > std::numeric_limits<unsigned>::max()) {
    return {ErrorCode::invalid_argument,
            "gfx12 FP16 GEMM requires tiled buffers, N divisible by 16, K divisible by 128, "
            "and batch in [1,32]"};
  }
  const dim3 grid(static_cast<unsigned>(rows / 16));
  const dim3 threads(batch > 16 ? 64U : 32U);
  if (format == Quantization::w2g128) {
    hipLaunchKernelGGL((gfx12_f16_gemm<2>), grid, threads, 0, stream,
                       tiled_weights, tiled_weight_scales, activations,
                       output, rows, columns, batch);
  } else if (format == Quantization::w4g128) {
    hipLaunchKernelGGL((gfx12_f16_gemm<4>), grid, threads, 0, stream,
                       tiled_weights, tiled_weight_scales, activations,
                       output, rows, columns, batch);
  } else {
    return {ErrorCode::unsupported, "gfx12 FP16 GEMM supports W2G128 and W4G128"};
  }
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "gfx12 FP16 GEMM launch failed");
  }
  return Status::ok_status();
}

}  // namespace gfxinfer
