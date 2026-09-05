// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/qwen_ops.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>
#include <string>
#include <string_view>

#include "gfxinfer/device.h"

namespace gfxinfer {
namespace {

constexpr unsigned kWaveSize = 32;

__device__ __forceinline__ float wave_sum(float value) {
  value += __shfl_down(value, 16, kWaveSize);
  value += __shfl_down(value, 8, kWaveSize);
  value += __shfl_down(value, 4, kWaveSize);
  value += __shfl_down(value, 2, kWaveSize);
  value += __shfl_down(value, 1, kWaveSize);
  return value;
}

__device__ __forceinline__ float bf16_to_float(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__global__ __launch_bounds__(256) void rms_norm(
    const __half* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    __half* __restrict__ output,
    std::size_t elements,
    float epsilon,
    bool add_unit_offset) {
  const auto row_offset = static_cast<std::size_t>(blockIdx.x) * elements;
  __shared__ float wave_sums[8];
  __shared__ float inverse_rms;
  float square_sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
    const float value = __half2float(input[row_offset + index]);
    square_sum = fmaf(value, value, square_sum);
  }
  square_sum = wave_sum(square_sum);
  const unsigned lane = threadIdx.x & (kWaveSize - 1U);
  const unsigned wave = threadIdx.x / kWaveSize;
  if (lane == 0) wave_sums[wave] = square_sum;
  __syncthreads();
  if (wave == 0) {
    float block_sum = lane < 8 ? wave_sums[lane] : 0.0F;
    block_sum = wave_sum(block_sum);
    if (lane == 0) inverse_rms = rsqrtf(block_sum / static_cast<float>(elements) + epsilon);
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
    float scale = bf16_to_float(weight[index]);
    if (add_unit_offset) scale += 1.0F;
    output[row_offset + index] = __float2half(
        __half2float(input[row_offset + index]) * inverse_rms * scale);
  }
}

__global__ __launch_bounds__(256) void dual_rms_norm_concat(
    const __half* __restrict__ left,
    const std::uint16_t* __restrict__ left_weight,
    const __half* __restrict__ right,
    const std::uint16_t* __restrict__ right_weight,
    __half* __restrict__ output,
    std::size_t elements,
    float epsilon,
    bool add_unit_offset) {
  const bool select_right = blockIdx.x != 0;
  const __half* input = select_right ? right : left;
  const std::uint16_t* weight =
      select_right ? right_weight : left_weight;
  output += select_right ? elements : 0;
  __shared__ float wave_sums[8];
  __shared__ float inverse_rms;
  float square_sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
    const float value = __half2float(input[index]);
    square_sum = fmaf(value, value, square_sum);
  }
  square_sum = wave_sum(square_sum);
  const unsigned lane = threadIdx.x & (kWaveSize - 1U);
  const unsigned wave = threadIdx.x / kWaveSize;
  if (lane == 0) wave_sums[wave] = square_sum;
  __syncthreads();
  if (wave == 0) {
    float block_sum = lane < 8 ? wave_sums[lane] : 0.0F;
    block_sum = wave_sum(block_sum);
    if (lane == 0) {
      inverse_rms = rsqrtf(
          block_sum / static_cast<float>(elements) + epsilon);
    }
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
    float scale = bf16_to_float(weight[index]);
    if (add_unit_offset) scale += 1.0F;
    output[index] = __float2half(
        __half2float(input[index]) * inverse_rms * scale);
  }
}

__global__ void silu_multiply(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    __half* __restrict__ output,
    std::size_t elements) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = gate[index];
  const float activated = value / (1.0F + expf(-value));
  output[index] = __float2half(activated * up[index]);
}

__global__ __launch_bounds__(128) void silu_multiply_quantize_i4(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    std::uint8_t* __restrict__ packed_output,
    __half* __restrict__ scales,
    std::size_t elements_per_row,
    std::size_t groups_per_row) {
  __shared__ float reduction[128];
  __shared__ __half rounded[128];
  const std::size_t linear_group = blockIdx.x;
  const std::size_t batch = linear_group / groups_per_row;
  const std::size_t group = linear_group - batch * groups_per_row;
  const std::size_t source =
      batch * elements_per_row + group * 128U + threadIdx.x;
  const float value = gate[source];
  const __half product = __float2half(
      (value / (1.0F + expf(-value))) * up[source]);
  rounded[threadIdx.x] = product;
  reduction[threadIdx.x] = fabsf(__half2float(product));
  __syncthreads();
  for (unsigned stride = 64; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = fmaxf(
          reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scales[linear_group] = __float2half(
        reduction[0] == 0.0F ? 1.0F : reduction[0] / 7.0F);
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    const float scale = __half2float(scales[linear_group]);
    int low = __float2int_rn(
        __half2float(rounded[threadIdx.x * 2U]) / scale);
    int high = __float2int_rn(
        __half2float(rounded[threadIdx.x * 2U + 1U]) / scale);
    low = max(-7, min(7, low));
    high = max(-7, min(7, high));
    packed_output[batch * (elements_per_row / 2U) + group * 64U + threadIdx.x] =
        static_cast<std::uint8_t>((low & 0x0f) | ((high & 0x0f) << 4U));
  }
}

__global__ void residual_add(
    const __half* __restrict__ residual,
    const float* __restrict__ update,
    __half* __restrict__ output,
    std::size_t elements) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2half(__half2float(residual[index]) + update[index]);
  }
}

__global__ __launch_bounds__(256) void residual_add_rms_norm(
    const __half* __restrict__ residual,
    const float* __restrict__ update,
    __half* __restrict__ residual_output,
    const std::uint16_t* __restrict__ weight,
    __half* __restrict__ normalized_output,
    std::size_t elements,
    float epsilon,
    bool add_unit_offset) {
  const auto row_offset = static_cast<std::size_t>(blockIdx.x) * elements;
  __shared__ float wave_sums[8];
  __shared__ float inverse_rms;
  float square_sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
    const auto offset = row_offset + index;
    const __half rounded = __float2half(
        __half2float(residual[offset]) + update[offset]);
    residual_output[offset] = rounded;
    const float value = __half2float(rounded);
    square_sum = fmaf(value, value, square_sum);
  }
  square_sum = wave_sum(square_sum);
  const unsigned lane = threadIdx.x & (kWaveSize - 1U);
  const unsigned wave = threadIdx.x / kWaveSize;
  if (lane == 0) wave_sums[wave] = square_sum;
  __syncthreads();
  if (wave == 0) {
    float block_sum = lane < 8 ? wave_sums[lane] : 0.0F;
    block_sum = wave_sum(block_sum);
    if (lane == 0) {
      inverse_rms = rsqrtf(
          block_sum / static_cast<float>(elements) + epsilon);
    }
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
    float scale = bf16_to_float(weight[index]);
    if (add_unit_offset) scale += 1.0F;
    normalized_output[row_offset + index] = __float2half(
        __half2float(residual_output[row_offset + index]) * inverse_rms * scale);
  }
}

__global__ void gdn_conv_update(
    const float* __restrict__ projected,
    const std::uint16_t* __restrict__ weight,
    const __half* __restrict__ initial_state,
    __half* __restrict__ final_state,
    __half* __restrict__ output,
    __half* __restrict__ appended_log,
    std::size_t channels,
    unsigned batch) {
  constexpr std::size_t width = 4;
  const auto channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  const auto* initial_channel_state = initial_state + channel * width;
  __half channel_state[width];
#pragma unroll
  for (std::size_t tap = 0; tap < width; ++tap) {
    channel_state[tap] = initial_channel_state[tap];
  }
  for (unsigned token = 0; token < batch; ++token) {
    channel_state[0] = channel_state[1];
    channel_state[1] = channel_state[2];
    channel_state[2] = channel_state[3];
    channel_state[3] = __float2half(
        projected[static_cast<std::size_t>(token) * channels + channel]);
    if (appended_log != nullptr) {
      appended_log[static_cast<std::size_t>(token) * channels + channel] =
          channel_state[3];
    }
    float convolution = 0.0F;
#pragma unroll
    for (std::size_t tap = 0; tap < width; ++tap) {
      convolution = fmaf(__half2float(channel_state[tap]),
                         bf16_to_float(weight[channel * width + tap]), convolution);
    }
    output[static_cast<std::size_t>(token) * channels + channel] =
        __float2half(convolution / (1.0F + expf(-convolution)));
  }
  auto* final_channel_state = final_state + channel * width;
#pragma unroll
  for (std::size_t tap = 0; tap < width; ++tap) {
    final_channel_state[tap] = channel_state[tap];
  }
}

__global__ __launch_bounds__(256) void gdn_recurrent_decode(
    const __half* __restrict__ qkv,
    const float* __restrict__ z,
    const float* __restrict__ projected_a,
    const float* __restrict__ projected_b,
    const std::uint16_t* __restrict__ a_log,
    const std::uint16_t* __restrict__ dt_bias,
    const std::uint16_t* __restrict__ norm_weight,
    const float* __restrict__ initial_states,
    float* __restrict__ final_states,
    __half* __restrict__ output,
    float* __restrict__ key_log,
    float* __restrict__ delta_log,
    float* __restrict__ decay_log,
    unsigned batch) {
  constexpr unsigned head_size = 128;
  constexpr unsigned key_heads = 16;
  constexpr unsigned key_elements = key_heads * head_size;
  constexpr unsigned value_heads = 48;
  constexpr unsigned value_elements = value_heads * head_size;
  const unsigned value_head = blockIdx.x;
  const unsigned key_head = value_head / 3U;
  const unsigned thread = threadIdx.x;
  __shared__ float query[head_size];
  __shared__ float key[head_size];
  __shared__ float delta[head_size];
  __shared__ float mixed[head_size];
  __shared__ float reductions[8];
  __shared__ float inverse_query_norm;
  __shared__ float inverse_key_norm;
  __shared__ float decay;
  __shared__ float beta;
  __shared__ float inverse_output_rms;

  for (unsigned token = 0; token < batch; ++token) {
    const auto* token_qkv = qkv + static_cast<std::size_t>(token) * 10240U;
    const auto* token_z = z + static_cast<std::size_t>(token) * 6144U;
    const auto* token_a = projected_a + static_cast<std::size_t>(token) * 48U;
    const auto* token_b = projected_b + static_cast<std::size_t>(token) * 48U;
    auto* token_output = output + static_cast<std::size_t>(token) * 6144U;
    float query_square = 0.0F;
    float key_square = 0.0F;
    if (thread < head_size) {
      const float q = __half2float(token_qkv[key_head * head_size + thread]);
      const float k = __half2float(
          token_qkv[key_elements + key_head * head_size + thread]);
    query[thread] = q;
    key[thread] = k;
    query_square = q * q;
    key_square = k * k;
  }
  query_square = wave_sum(query_square);
  key_square = wave_sum(key_square);
  const unsigned lane = thread & (kWaveSize - 1U);
  const unsigned wave = thread / kWaveSize;
  if (lane == 0) reductions[wave] = query_square;
  __syncthreads();
  if (wave == 0) {
    float sum = lane < 8 ? reductions[lane] : 0.0F;
    sum = wave_sum(sum);
    if (lane == 0) inverse_query_norm = rsqrtf(sum + 1.0e-6F) * 0.08838834764831845F;
  }
  __syncthreads();
  if (lane == 0) reductions[wave] = key_square;
  __syncthreads();
  if (wave == 0) {
    float sum = lane < 8 ? reductions[lane] : 0.0F;
    sum = wave_sum(sum);
    if (lane == 0) inverse_key_norm = rsqrtf(sum + 1.0e-6F);
  }
  __syncthreads();
    if (thread == 0) {
      const float a_value = token_a[value_head] + bf16_to_float(dt_bias[value_head]);
    const float softplus = a_value > 20.0F ? a_value : log1pf(expf(a_value));
    const float g = -expf(bf16_to_float(a_log[value_head])) * softplus;
    decay = expf(g);
      const float b_value = token_b[value_head];
    beta = 1.0F / (1.0F + expf(-b_value));
  }
  if (thread < head_size) {
    query[thread] *= inverse_query_norm;
    key[thread] *= inverse_key_norm;
  }
  __syncthreads();
  if (decay_log != nullptr && thread == 0) {
    decay_log[static_cast<std::size_t>(token) * value_heads + value_head] = decay;
  }
  if (key_log != nullptr && value_head % 3U == 0 && thread < head_size) {
    key_log[static_cast<std::size_t>(token) * key_elements +
            key_head * head_size + thread] = key[thread];
  }

  const auto state_offset =
      static_cast<std::size_t>(value_head) * head_size * head_size;
  const auto* initial_state = initial_states + state_offset;
  auto* state = final_states + state_offset;
  for (unsigned index = thread; index < head_size * head_size; index += blockDim.x) {
    const float previous = token == 0 ? initial_state[index] : state[index];
    state[index] = previous * decay;
  }
  __syncthreads();
  if (thread < head_size) {
    float memory = 0.0F;
    for (unsigned k = 0; k < head_size; ++k) {
      memory = fmaf(state[k * head_size + thread], key[k], memory);
    }
    const float value = __half2float(
          token_qkv[2U * key_elements + value_head * head_size + thread]);
    delta[thread] = (value - memory) * beta;
    if (delta_log != nullptr) {
      delta_log[static_cast<std::size_t>(token) * value_elements +
                value_head * head_size + thread] = delta[thread];
    }
  }
  __syncthreads();
  for (unsigned index = thread; index < head_size * head_size; index += blockDim.x) {
    const unsigned k = index / head_size;
    const unsigned v = index % head_size;
    state[index] = fmaf(key[k], delta[v], state[index]);
  }
  __syncthreads();
  float output_square = 0.0F;
  if (thread < head_size) {
    float value = 0.0F;
    for (unsigned k = 0; k < head_size; ++k) {
      value = fmaf(state[k * head_size + thread], query[k], value);
    }
    mixed[thread] = value;
    output_square = value * value;
  }
  output_square = wave_sum(output_square);
  if (lane == 0) reductions[wave] = output_square;
  __syncthreads();
  if (wave == 0) {
    float sum = lane < 8 ? reductions[lane] : 0.0F;
    sum = wave_sum(sum);
    if (lane == 0) inverse_output_rms = rsqrtf(sum / head_size + 1.0e-6F);
  }
  __syncthreads();
  if (thread < head_size) {
    const auto index = value_head * head_size + thread;
      const float gate_value = token_z[index];
    const float gate = gate_value / (1.0F + expf(-gate_value));
    const float normalized = mixed[thread] * inverse_output_rms *
                             bf16_to_float(norm_weight[thread]);
      token_output[index] = __float2half(normalized * gate);
    }
  }
}

__global__ void gdn_restore_conv_prefix(
    __half* __restrict__ state,
    const __half* __restrict__ appended_log,
    std::size_t channels,
    unsigned accepted_tokens) {
  constexpr std::size_t width = 4;
  const auto channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  auto* channel_state = state + channel * width;
  for (unsigned token = 0; token < accepted_tokens; ++token) {
    channel_state[0] = channel_state[1];
    channel_state[1] = channel_state[2];
    channel_state[2] = channel_state[3];
    channel_state[3] =
        appended_log[static_cast<std::size_t>(token) * channels + channel];
  }
}

__global__ __launch_bounds__(256) void gdn_restore_recurrent_prefix(
    float* __restrict__ states,
    const float* __restrict__ key_log,
    const float* __restrict__ delta_log,
    const float* __restrict__ decay_log,
    unsigned accepted_tokens) {
  constexpr unsigned head_size = 128;
  constexpr unsigned key_elements = 16 * head_size;
  constexpr unsigned value_heads = 48;
  constexpr unsigned value_elements = value_heads * head_size;
  const unsigned value_head = blockIdx.x;
  const unsigned key_head = value_head / 3U;
  const unsigned thread = threadIdx.x;
  auto* state = states + static_cast<std::size_t>(value_head) * head_size * head_size;
  for (unsigned token = 0; token < accepted_tokens; ++token) {
    const float decay =
        decay_log[static_cast<std::size_t>(token) * value_heads + value_head];
    for (unsigned index = thread; index < head_size * head_size;
         index += blockDim.x) {
      state[index] *= decay;
    }
    __syncthreads();
    for (unsigned index = thread; index < head_size * head_size;
         index += blockDim.x) {
      const unsigned k = index / head_size;
      const unsigned v = index % head_size;
      const float key = key_log[static_cast<std::size_t>(token) * key_elements +
                                key_head * head_size + k];
      const float delta = delta_log[static_cast<std::size_t>(token) * value_elements +
                                    value_head * head_size + v];
      state[index] = fmaf(key, delta, state[index]);
    }
    __syncthreads();
  }
}

__global__ void gdn_restore_conv_prefix_all(
    __half* __restrict__ states,
    const __half* __restrict__ appended_logs,
    std::size_t channels,
    unsigned maximum_logged_tokens,
    unsigned layer_count,
    unsigned accepted_tokens) {
  constexpr std::size_t width = 4;
  const auto linear_channel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total_channels = static_cast<std::size_t>(layer_count) * channels;
  if (linear_channel >= total_channels) return;
  const auto layer = linear_channel / channels;
  const auto channel = linear_channel - layer * channels;
  auto* channel_state =
      states + layer * channels * width + channel * width;
  const auto* appended_log =
      appended_logs + layer * maximum_logged_tokens * channels;
  for (unsigned token = 0; token < accepted_tokens; ++token) {
    channel_state[0] = channel_state[1];
    channel_state[1] = channel_state[2];
    channel_state[2] = channel_state[3];
    channel_state[3] =
        appended_log[static_cast<std::size_t>(token) * channels + channel];
  }
}

__global__ __launch_bounds__(256) void gdn_restore_recurrent_prefix_all(
    float* __restrict__ states,
    const float* __restrict__ key_logs,
    const float* __restrict__ delta_logs,
    const float* __restrict__ decay_logs,
    unsigned maximum_logged_tokens,
    unsigned accepted_tokens) {
  constexpr unsigned head_size = 128;
  constexpr unsigned key_elements = 16 * head_size;
  constexpr unsigned value_heads = 48;
  constexpr unsigned value_elements = value_heads * head_size;
  constexpr std::size_t state_elements =
      static_cast<std::size_t>(value_heads) * head_size * head_size;
  const unsigned layer = blockIdx.x / value_heads;
  const unsigned value_head = blockIdx.x % value_heads;
  const unsigned key_head = value_head / 3U;
  const unsigned thread = threadIdx.x;
  auto* state = states + static_cast<std::size_t>(layer) * state_elements +
                static_cast<std::size_t>(value_head) * head_size * head_size;
  const auto* key_log =
      key_logs + static_cast<std::size_t>(layer) * maximum_logged_tokens * key_elements;
  const auto* delta_log =
      delta_logs + static_cast<std::size_t>(layer) * maximum_logged_tokens * value_elements;
  const auto* decay_log =
      decay_logs + static_cast<std::size_t>(layer) * maximum_logged_tokens * value_heads;
  for (unsigned token = 0; token < accepted_tokens; ++token) {
    const float decay =
        decay_log[static_cast<std::size_t>(token) * value_heads + value_head];
    for (unsigned index = thread; index < head_size * head_size;
         index += blockDim.x) {
      state[index] *= decay;
    }
    __syncthreads();
    for (unsigned index = thread; index < head_size * head_size;
         index += blockDim.x) {
      const unsigned k = index / head_size;
      const unsigned v = index % head_size;
      const float key = key_log[static_cast<std::size_t>(token) * key_elements +
                                key_head * head_size + k];
      const float delta = delta_log[static_cast<std::size_t>(token) * value_elements +
                                    value_head * head_size + v];
      state[index] = fmaf(key, delta, state[index]);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(256) void head_rms_norm_rope(
    const float* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    __half* __restrict__ output,
    std::size_t head_size,
    std::size_t head_count,
    std::size_t input_stride,
    std::size_t rotary_size,
    std::uint64_t position,
    float rope_theta) {
  const auto token = static_cast<std::size_t>(blockIdx.x) / head_count;
  const auto head = static_cast<std::size_t>(blockIdx.x) % head_count;
  input += token * head_count * input_stride;
  output += token * head_count * head_size;
  position += token;
  const unsigned thread = threadIdx.x;
  __shared__ float normalized[256];
  __shared__ float reductions[8];
  __shared__ float inverse_rms;
  float square = 0.0F;
  if (thread < head_size) {
    const float value = input[head * input_stride + thread];
    normalized[thread] = value;
    square = value * value;
  }
  square = wave_sum(square);
  const unsigned lane = thread & (kWaveSize - 1U);
  const unsigned wave = thread / kWaveSize;
  if (lane == 0) reductions[wave] = square;
  __syncthreads();
  if (wave == 0) {
    float sum = lane < 8 ? reductions[lane] : 0.0F;
    sum = wave_sum(sum);
    if (lane == 0) inverse_rms = rsqrtf(sum / static_cast<float>(head_size) + 1.0e-6F);
  }
  __syncthreads();
  if (thread < head_size) {
    normalized[thread] *= inverse_rms * (1.0F + bf16_to_float(weight[thread]));
  }
  __syncthreads();
  if (thread < head_size) {
    float value = normalized[thread];
    if (thread < rotary_size) {
      const auto half_rotary = static_cast<unsigned>(rotary_size / 2U);
      const unsigned frequency_index = thread % half_rotary;
      const float exponent = (2.0F * static_cast<float>(frequency_index)) /
                             static_cast<float>(rotary_size);
      const float inverse_frequency = powf(rope_theta, -exponent);
      const float angle = static_cast<float>(position) * inverse_frequency;
      const float partner = normalized[thread < half_rotary ? thread + half_rotary
                                                             : thread - half_rotary];
      const float rotated = thread < half_rotary ? -partner : partner;
      value = fmaf(value, cosf(angle), rotated * sinf(angle));
    }
    output[head * head_size + thread] = __float2half(value);
  }
}

__global__ __launch_bounds__(256) void qk_rms_norm_rope(
    const float* __restrict__ query_input,
    const std::uint16_t* __restrict__ query_weight,
    __half* __restrict__ query_output,
    std::size_t query_head_count,
    std::size_t query_input_stride,
    const float* __restrict__ key_input,
    const std::uint16_t* __restrict__ key_weight,
    __half* __restrict__ key_output,
    std::size_t key_head_count,
    std::size_t key_input_stride,
    std::size_t head_size,
    std::size_t rotary_size,
    std::uint64_t base_position,
    const std::uint64_t* __restrict__ indirect_base_position,
    float rope_theta) {
  if (indirect_base_position != nullptr) {
    base_position = *indirect_base_position;
  }
  const auto heads_per_token = query_head_count + key_head_count;
  const auto token = static_cast<std::size_t>(blockIdx.x) / heads_per_token;
  const auto local_head = static_cast<std::size_t>(blockIdx.x) % heads_per_token;
  const bool is_key = local_head >= query_head_count;
  const auto head = is_key ? local_head - query_head_count : local_head;
  const auto head_count = is_key ? key_head_count : query_head_count;
  const auto input_stride = is_key ? key_input_stride : query_input_stride;
  const float* input = is_key ? key_input : query_input;
  const std::uint16_t* weight = is_key ? key_weight : query_weight;
  __half* output = is_key ? key_output : query_output;
  input += token * head_count * input_stride;
  output += token * head_count * head_size;
  const auto position = base_position + token;
  const unsigned thread = threadIdx.x;
  __shared__ float normalized[256];
  __shared__ float reductions[8];
  __shared__ float inverse_rms;
  float square = 0.0F;
  if (thread < head_size) {
    const float value = input[head * input_stride + thread];
    normalized[thread] = value;
    square = value * value;
  }
  square = wave_sum(square);
  const unsigned lane = thread & (kWaveSize - 1U);
  const unsigned wave = thread / kWaveSize;
  if (lane == 0) reductions[wave] = square;
  __syncthreads();
  if (wave == 0) {
    float sum = lane < 8 ? reductions[lane] : 0.0F;
    sum = wave_sum(sum);
    if (lane == 0) {
      inverse_rms = rsqrtf(sum / static_cast<float>(head_size) + 1.0e-6F);
    }
  }
  __syncthreads();
  if (thread < head_size) {
    normalized[thread] *=
        inverse_rms * (1.0F + bf16_to_float(weight[thread]));
  }
  __syncthreads();
  if (thread < head_size) {
    float value = normalized[thread];
    if (thread < rotary_size) {
      const auto half_rotary = static_cast<unsigned>(rotary_size / 2U);
      const unsigned frequency_index = thread % half_rotary;
      const float exponent =
          (2.0F * static_cast<float>(frequency_index)) /
          static_cast<float>(rotary_size);
      const float inverse_frequency = powf(rope_theta, -exponent);
      const float angle = static_cast<float>(position) * inverse_frequency;
      const float partner = normalized[
          thread < half_rotary ? thread + half_rotary : thread - half_rotary];
      const float rotated = thread < half_rotary ? -partner : partner;
      value = fmaf(value, cosf(angle), rotated * sinf(angle));
    }
    output[head * head_size + thread] = __float2half(value);
  }
}

__global__ void kv_append(
    const __half* __restrict__ key,
    const float* __restrict__ value,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    std::size_t position,
    const std::uint64_t* __restrict__ indirect_position,
    std::size_t capacity,
    unsigned batch) {
  if (indirect_position != nullptr) position = *indirect_position;
  constexpr std::size_t elements = 4 * 256;
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(batch) * elements) return;
  const auto token = index / elements;
  const auto item = index % elements;
  key_cache[(position + token) * elements + item] = key[index];
  value_cache[(position + token) * elements + item] = __float2half(value[index]);
  (void)capacity;
}

__global__ __launch_bounds__(256) void gated_attention_decode(
    const __half* __restrict__ query,
    const float* __restrict__ gate,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    __half* __restrict__ output,
    std::size_t base_token_count,
    const std::uint64_t* __restrict__ indirect_base_position,
    std::size_t capacity,
    unsigned batch) {
  constexpr unsigned head_size = 256;
  constexpr unsigned query_heads = 24;
  constexpr unsigned key_value_heads = 4;
  constexpr unsigned group_size = query_heads / key_value_heads;
  const unsigned token_index = blockIdx.x / query_heads;
  const unsigned query_head = blockIdx.x % query_heads;
  const unsigned kv_head = query_head / group_size;
  const unsigned thread = threadIdx.x;
  const unsigned lane = thread & (kWaveSize - 1U);
  const unsigned wave = thread / kWaveSize;
  __shared__ float reductions[8];
  __shared__ float alpha;
  __shared__ float beta;
  __shared__ float denominator;
  __shared__ float maximum;
  if (indirect_base_position != nullptr) {
    base_token_count = static_cast<std::size_t>(*indirect_base_position) + 1U;
  }
  const std::size_t token_count = base_token_count + token_index;
  query += static_cast<std::size_t>(token_index) * query_heads * head_size;
  gate += static_cast<std::size_t>(token_index) * query_heads * head_size * 2U;
  output += static_cast<std::size_t>(token_index) * query_heads * head_size;
  float accumulator = 0.0F;
  if (thread == 0) {
    denominator = 0.0F;
    maximum = -INFINITY;
  }
  __syncthreads();
  const float q = __half2float(query[query_head * head_size + thread]);
  for (std::size_t token = 0; token < token_count; ++token) {
    const auto cache_index = (token * key_value_heads + kv_head) * head_size + thread;
    float score = q * __half2float(key_cache[cache_index]);
    score = wave_sum(score);
    if (lane == 0) reductions[wave] = score;
    __syncthreads();
    if (wave == 0) {
      float sum = lane < 8 ? reductions[lane] : 0.0F;
      sum = wave_sum(sum);
      if (lane == 0) {
        sum *= 0.0625F;
        const float next_maximum = fmaxf(maximum, sum);
        alpha = expf(maximum - next_maximum);
        beta = expf(sum - next_maximum);
        denominator = denominator * alpha + beta;
        maximum = next_maximum;
      }
    }
    __syncthreads();
    accumulator = accumulator * alpha +
                  __half2float(value_cache[cache_index]) * beta;
    __syncthreads();
  }
  const auto output_index = query_head * head_size + thread;
  const float gate_value = gate[query_head * (head_size * 2U) + head_size + thread];
  const float gate_scale = 1.0F / (1.0F + expf(-gate_value));
  output[output_index] = __float2half(accumulator / denominator * gate_scale);
  (void)batch;
  (void)capacity;
}

__global__ __launch_bounds__(256) void argmax_logits(
    const float* __restrict__ logits,
    const std::uint32_t* __restrict__ token_ids,
    std::uint32_t* __restrict__ token,
    std::size_t vocabulary_size,
    std::size_t row_stride) {
  logits += static_cast<std::size_t>(blockIdx.x) * row_stride;
  token += blockIdx.x;
  __shared__ float maximum_values[256];
  __shared__ std::uint32_t maximum_indices[256];
  float maximum = -INFINITY;
  std::uint32_t maximum_index = 0;
  for (std::size_t index = threadIdx.x; index < vocabulary_size; index += blockDim.x) {
    const float value = logits[index];
    if (value > maximum || (value == maximum && index < maximum_index)) {
      maximum = value;
      maximum_index = static_cast<std::uint32_t>(index);
    }
  }
  maximum_values[threadIdx.x] = maximum;
  maximum_indices[threadIdx.x] = maximum_index;
  __syncthreads();
  for (unsigned stride = 128; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const float other_value = maximum_values[threadIdx.x + stride];
      const auto other_index = maximum_indices[threadIdx.x + stride];
      if (other_value > maximum_values[threadIdx.x] ||
          (other_value == maximum_values[threadIdx.x] &&
           other_index < maximum_indices[threadIdx.x])) {
        maximum_values[threadIdx.x] = other_value;
        maximum_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const auto index = maximum_indices[0];
    token[0] = token_ids == nullptr ? index : token_ids[index];
  }
}

__global__ void float_to_half(
    const float* __restrict__ input,
    __half* __restrict__ output,
    std::size_t elements) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = __float2half(input[index]);
}

__global__ void concat_half(
    const __half* __restrict__ left,
    std::size_t left_elements,
    const __half* __restrict__ right,
    std::size_t right_elements,
    __half* __restrict__ output) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = left_elements + right_elements;
  if (index < total) {
    output[index] = index < left_elements ? left[index] : right[index - left_elements];
  }
}

[[nodiscard]] Status elementwise_launch_status(
    const void* first,
    const void* second,
    const void* output,
    std::size_t elements,
    std::string_view context) {
  if (first == nullptr || second == nullptr || output == nullptr || elements == 0 ||
      elements > std::numeric_limits<unsigned>::max() * 256ULL) {
    return {ErrorCode::invalid_argument, "invalid " + std::string(context) + " arguments"};
  }
  return Status::ok_status();
}

}  // namespace

Status launch_rms_norm(
    const __half* input,
    const std::uint16_t* bf16_weight,
    __half* output,
    std::size_t elements,
    float epsilon,
    bool add_unit_offset,
    hipStream_t stream) {
  if (input == nullptr || bf16_weight == nullptr || output == nullptr ||
      elements == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return {ErrorCode::invalid_argument, "invalid RMSNorm arguments"};
  }
  hipLaunchKernelGGL(rms_norm, dim3(1), dim3(256), 0, stream,
                     input, bf16_weight, output, elements, epsilon, add_unit_offset);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "RMSNorm launch failed");
  }
  return Status::ok_status();
}

Status launch_rms_norm_batch(
    const __half* input,
    const std::uint16_t* bf16_weight,
    __half* output,
    std::size_t elements,
    unsigned batch,
    float epsilon,
    bool add_unit_offset,
    hipStream_t stream) {
  if (input == nullptr || bf16_weight == nullptr || output == nullptr ||
      elements == 0 || batch == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return {ErrorCode::invalid_argument, "invalid batched RMSNorm arguments"};
  }
  hipLaunchKernelGGL(rms_norm, dim3(batch), dim3(256), 0, stream,
                     input, bf16_weight, output, elements, epsilon, add_unit_offset);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "batched RMSNorm launch failed");
  }
  return Status::ok_status();
}

Status launch_dual_rms_norm_concat(
    const __half* left,
    const std::uint16_t* left_bf16_weight,
    const __half* right,
    const std::uint16_t* right_bf16_weight,
    __half* concatenated_output,
    std::size_t elements,
    float epsilon,
    bool add_unit_offset,
    hipStream_t stream) {
  if (left == nullptr || left_bf16_weight == nullptr || right == nullptr ||
      right_bf16_weight == nullptr || concatenated_output == nullptr ||
      elements == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return {ErrorCode::invalid_argument,
            "invalid dual RMSNorm/concat arguments"};
  }
  hipLaunchKernelGGL(
      dual_rms_norm_concat, dim3(2), dim3(256), 0, stream,
      left, left_bf16_weight, right, right_bf16_weight,
      concatenated_output, elements, epsilon, add_unit_offset);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "dual RMSNorm/concat launch failed");
  }
  return Status::ok_status();
}

Status launch_silu_multiply(
    const float* gate,
    const float* up,
    __half* output,
    std::size_t elements,
    hipStream_t stream) {
  auto status = elementwise_launch_status(gate, up, output, elements, "SiLU multiply");
  if (!status.ok()) return status;
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(silu_multiply,
                     dim3(static_cast<unsigned>((elements + threads - 1U) / threads)),
                     dim3(threads), 0, stream, gate, up, output, elements);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "SiLU multiply launch failed");
  }
  return Status::ok_status();
}

Status launch_silu_multiply_quantize_i4(
    const float* gate,
    const float* up,
    std::uint8_t* packed_output,
    __half* scales,
    std::size_t elements_per_row,
    unsigned batch,
    hipStream_t stream) {
  if (gate == nullptr || up == nullptr || packed_output == nullptr ||
      scales == nullptr || elements_per_row == 0 ||
      elements_per_row % 128 != 0 || batch == 0 || batch > 32) {
    return {ErrorCode::invalid_argument,
            "invalid fused SiLU/A4 quantization arguments"};
  }
  const auto groups = elements_per_row / 128;
  hipLaunchKernelGGL(
      silu_multiply_quantize_i4,
      dim3(static_cast<unsigned>(groups * batch)), dim3(128), 0, stream,
      gate, up, packed_output, scales, elements_per_row, groups);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "fused SiLU/A4 quantization launch failed");
  }
  return Status::ok_status();
}

Status launch_residual_add(
    const __half* residual,
    const float* update,
    __half* output,
    std::size_t elements,
    hipStream_t stream) {
  auto status = elementwise_launch_status(residual, update, output, elements, "residual add");
  if (!status.ok()) return status;
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(residual_add,
                     dim3(static_cast<unsigned>((elements + threads - 1U) / threads)),
                     dim3(threads), 0, stream, residual, update, output, elements);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "residual add launch failed");
  }
  return Status::ok_status();
}

Status launch_residual_add_rms_norm_batch(
    const __half* residual,
    const float* update,
    __half* residual_output,
    const std::uint16_t* bf16_weight,
    __half* normalized_output,
    std::size_t elements,
    unsigned batch,
    float epsilon,
    bool add_unit_offset,
    hipStream_t stream) {
  if (residual == nullptr || update == nullptr || residual_output == nullptr ||
      bf16_weight == nullptr || normalized_output == nullptr || elements == 0 ||
      batch == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return {ErrorCode::invalid_argument,
            "invalid fused residual/RMSNorm arguments"};
  }
  hipLaunchKernelGGL(residual_add_rms_norm, dim3(batch), dim3(256), 0, stream,
                     residual, update, residual_output, bf16_weight,
                     normalized_output, elements, epsilon, add_unit_offset);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "fused residual/RMSNorm launch failed");
  }
  return Status::ok_status();
}

Status launch_gdn_conv_update(
    const float* projected_qkv,
    const std::uint16_t* bf16_conv_weight,
    __half* conv_state,
    __half* convolved_qkv,
    hipStream_t stream) {
  if (projected_qkv == nullptr || bf16_conv_weight == nullptr || conv_state == nullptr ||
      convolved_qkv == nullptr) {
    return {ErrorCode::invalid_argument, "invalid GDN convolution arguments"};
  }
  constexpr unsigned channels = 10240;
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(gdn_conv_update, dim3((channels + threads - 1U) / threads),
                     dim3(threads), 0, stream, projected_qkv, bf16_conv_weight,
                     conv_state, conv_state, convolved_qkv, nullptr, channels, 1U);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "GDN convolution update launch failed");
  }
  return Status::ok_status();
}

Status launch_gdn_conv_update_block(
    const float* projected_qkv,
    const std::uint16_t* bf16_conv_weight,
    const __half* initial_conv_state,
    __half* final_conv_state,
    __half* convolved_qkv,
    __half* appended_log,
    std::size_t channels,
    unsigned batch,
    hipStream_t stream) {
  if (projected_qkv == nullptr || bf16_conv_weight == nullptr ||
      initial_conv_state == nullptr || final_conv_state == nullptr ||
      convolved_qkv == nullptr || channels == 0 || batch == 0 ||
      channels > std::numeric_limits<unsigned>::max() * 256ULL) {
    return {ErrorCode::invalid_argument, "invalid block GDN convolution arguments"};
  }
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(
      gdn_conv_update,
      dim3(static_cast<unsigned>((channels + threads - 1U) / threads)),
      dim3(threads), 0, stream, projected_qkv, bf16_conv_weight,
      initial_conv_state, final_conv_state, convolved_qkv, appended_log,
      channels, batch);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "block GDN convolution update launch failed");
  }
  return Status::ok_status();
}

Status launch_gdn_recurrent_decode(
    const __half* convolved_qkv,
    const float* projected_z,
    const float* projected_a,
    const float* projected_b,
    const std::uint16_t* bf16_a_log,
    const std::uint16_t* bf16_dt_bias,
    const std::uint16_t* bf16_norm_weight,
    float* recurrent_state,
    __half* output,
    hipStream_t stream) {
  if (convolved_qkv == nullptr || projected_z == nullptr || projected_a == nullptr ||
      projected_b == nullptr || bf16_a_log == nullptr || bf16_dt_bias == nullptr ||
      bf16_norm_weight == nullptr || recurrent_state == nullptr || output == nullptr) {
    return {ErrorCode::invalid_argument, "invalid GDN recurrent decode arguments"};
  }
  hipLaunchKernelGGL(gdn_recurrent_decode, dim3(48), dim3(256), 0, stream,
                     convolved_qkv, projected_z, projected_a, projected_b,
                     bf16_a_log, bf16_dt_bias, bf16_norm_weight,
                     recurrent_state, recurrent_state, output,
                     nullptr, nullptr, nullptr, 1U);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "GDN recurrent decode launch failed");
  }
  return Status::ok_status();
}

Status launch_gdn_recurrent_decode_block(
    const __half* convolved_qkv,
    const float* projected_z,
    const float* projected_a,
    const float* projected_b,
    const std::uint16_t* bf16_a_log,
    const std::uint16_t* bf16_dt_bias,
    const std::uint16_t* bf16_norm_weight,
    const float* initial_recurrent_state,
    float* final_recurrent_state,
    __half* output,
    float* key_log,
    float* delta_log,
    float* decay_log,
    unsigned batch,
    hipStream_t stream) {
  if (convolved_qkv == nullptr || projected_z == nullptr || projected_a == nullptr ||
      projected_b == nullptr || bf16_a_log == nullptr || bf16_dt_bias == nullptr ||
      bf16_norm_weight == nullptr || initial_recurrent_state == nullptr ||
      final_recurrent_state == nullptr || output == nullptr || batch == 0) {
    return {ErrorCode::invalid_argument, "invalid block GDN recurrent decode arguments"};
  }
  hipLaunchKernelGGL(gdn_recurrent_decode, dim3(48), dim3(256), 0, stream,
                     convolved_qkv, projected_z, projected_a, projected_b,
                     bf16_a_log, bf16_dt_bias, bf16_norm_weight,
                     initial_recurrent_state, final_recurrent_state, output,
                     key_log, delta_log, decay_log, batch);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "block GDN recurrent decode launch failed");
  }
  return Status::ok_status();
}

Status launch_gdn_restore_prefix(
    __half* conv_state,
    float* recurrent_state,
    const __half* appended_log,
    const float* key_log,
    const float* delta_log,
    const float* decay_log,
    std::size_t channels,
    unsigned accepted_tokens,
    hipStream_t stream) {
  if (conv_state == nullptr || recurrent_state == nullptr || appended_log == nullptr ||
      key_log == nullptr || delta_log == nullptr || decay_log == nullptr ||
      channels == 0 || accepted_tokens == 0) {
    return {ErrorCode::invalid_argument, "invalid GDN prefix restore arguments"};
  }
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(
      gdn_restore_conv_prefix,
      dim3(static_cast<unsigned>((channels + threads - 1U) / threads)),
      dim3(threads), 0, stream, conv_state, appended_log, channels, accepted_tokens);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "GDN convolution prefix restore launch failed");
  }
  hipLaunchKernelGGL(
      gdn_restore_recurrent_prefix, dim3(48), dim3(256), 0, stream,
      recurrent_state, key_log, delta_log, decay_log, accepted_tokens);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "GDN recurrent prefix restore launch failed");
  }
  return Status::ok_status();
}

Status launch_gdn_restore_prefix_all(
    __half* conv_states,
    float* recurrent_states,
    const __half* appended_logs,
    const float* key_logs,
    const float* delta_logs,
    const float* decay_logs,
    std::size_t channels,
    unsigned layer_count,
    unsigned maximum_logged_tokens,
    unsigned accepted_tokens,
    hipStream_t stream) {
  if (conv_states == nullptr || recurrent_states == nullptr ||
      appended_logs == nullptr || key_logs == nullptr || delta_logs == nullptr ||
      decay_logs == nullptr || channels == 0 || layer_count == 0 ||
      maximum_logged_tokens == 0 || accepted_tokens == 0 ||
      accepted_tokens > maximum_logged_tokens) {
    return {ErrorCode::invalid_argument,
            "invalid all-layer GDN prefix restore arguments"};
  }
  constexpr unsigned threads = 256;
  const auto total_channels = static_cast<std::size_t>(layer_count) * channels;
  hipLaunchKernelGGL(
      gdn_restore_conv_prefix_all,
      dim3(static_cast<unsigned>((total_channels + threads - 1U) / threads)),
      dim3(threads), 0, stream, conv_states, appended_logs, channels,
      maximum_logged_tokens, layer_count, accepted_tokens);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "all-layer GDN convolution prefix restore launch failed");
  }
  hipLaunchKernelGGL(
      gdn_restore_recurrent_prefix_all, dim3(layer_count * 48U), dim3(256), 0,
      stream, recurrent_states, key_logs, delta_logs, decay_logs,
      maximum_logged_tokens, accepted_tokens);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "all-layer GDN recurrent prefix restore launch failed");
  }
  return Status::ok_status();
}

Status launch_head_rms_norm_rope(
    const float* input,
    const std::uint16_t* bf16_weight,
    __half* output,
    std::size_t head_count,
    std::size_t head_size,
    std::size_t input_stride,
    std::size_t rotary_size,
    std::uint64_t position,
    float rope_theta,
    hipStream_t stream) {
  if (input == nullptr || bf16_weight == nullptr || output == nullptr ||
      head_count == 0 || head_count > std::numeric_limits<unsigned>::max() ||
      head_size == 0 || head_size > 256 || input_stride < head_size || rotary_size == 0 ||
      rotary_size > head_size || (rotary_size & 1U) != 0 ||
      !std::isfinite(rope_theta) || rope_theta <= 1.0F) {
    return {ErrorCode::invalid_argument, "invalid head RMSNorm/RoPE arguments"};
  }
  hipLaunchKernelGGL(head_rms_norm_rope, dim3(static_cast<unsigned>(head_count)),
                     dim3(256), 0, stream, input, bf16_weight, output, head_size,
                     head_count, input_stride, rotary_size, position, rope_theta);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "head RMSNorm/RoPE launch failed");
  }
  return Status::ok_status();
}

Status launch_head_rms_norm_rope_block(
    const float* input,
    const std::uint16_t* bf16_weight,
    __half* output,
    std::size_t head_count,
    std::size_t head_size,
    std::size_t input_stride,
    std::size_t rotary_size,
    std::uint64_t base_position,
    unsigned batch,
    float rope_theta,
    hipStream_t stream) {
  if (input == nullptr || bf16_weight == nullptr || output == nullptr ||
      head_count == 0 || batch == 0 ||
      head_count > std::numeric_limits<unsigned>::max() / batch ||
      head_size == 0 || head_size > 256 || input_stride < head_size ||
      rotary_size == 0 || rotary_size > head_size || (rotary_size & 1U) != 0 ||
      !std::isfinite(rope_theta) || rope_theta <= 1.0F) {
    return {ErrorCode::invalid_argument, "invalid block head RMSNorm/RoPE arguments"};
  }
  hipLaunchKernelGGL(
      head_rms_norm_rope,
      dim3(static_cast<unsigned>(head_count * batch)), dim3(256), 0, stream,
      input, bf16_weight, output, head_size, head_count, input_stride,
      rotary_size, base_position, rope_theta);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "block head RMSNorm/RoPE launch failed");
  }
  return Status::ok_status();
}

Status launch_qk_rms_norm_rope_block(
    const float* query_input,
    const std::uint16_t* query_bf16_weight,
    __half* query_output,
    std::size_t query_head_count,
    std::size_t query_input_stride,
    const float* key_input,
    const std::uint16_t* key_bf16_weight,
    __half* key_output,
    std::size_t key_head_count,
    std::size_t key_input_stride,
    std::size_t head_size,
    std::size_t rotary_size,
    std::uint64_t base_position,
    unsigned batch,
    float rope_theta,
    hipStream_t stream) {
  if (query_input == nullptr || query_bf16_weight == nullptr ||
      query_output == nullptr || key_input == nullptr ||
      key_bf16_weight == nullptr || key_output == nullptr ||
      query_head_count == 0 || key_head_count == 0 || batch == 0 ||
      head_size == 0 || head_size > 256 ||
      query_input_stride < head_size || key_input_stride < head_size ||
      rotary_size == 0 || rotary_size > head_size ||
      (rotary_size & 1U) != 0 || !std::isfinite(rope_theta) ||
      rope_theta <= 1.0F ||
      query_head_count + key_head_count >
          std::numeric_limits<unsigned>::max() / batch) {
    return {ErrorCode::invalid_argument,
            "invalid fused Q/K RMSNorm/RoPE arguments"};
  }
  hipLaunchKernelGGL(
      qk_rms_norm_rope,
      dim3(static_cast<unsigned>((query_head_count + key_head_count) * batch)),
      dim3(256), 0, stream, query_input, query_bf16_weight, query_output,
      query_head_count, query_input_stride, key_input, key_bf16_weight,
      key_output, key_head_count, key_input_stride, head_size, rotary_size,
      base_position, nullptr, rope_theta);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "fused Q/K RMSNorm/RoPE launch failed");
  }
  return Status::ok_status();
}

Status launch_qk_rms_norm_rope_block_indirect_position(
    const float* query_input,
    const std::uint16_t* query_bf16_weight,
    __half* query_output,
    std::size_t query_head_count,
    std::size_t query_input_stride,
    const float* key_input,
    const std::uint16_t* key_bf16_weight,
    __half* key_output,
    std::size_t key_head_count,
    std::size_t key_input_stride,
    std::size_t head_size,
    std::size_t rotary_size,
    const std::uint64_t* device_base_position,
    unsigned batch,
    float rope_theta,
    hipStream_t stream) {
  if (query_input == nullptr || query_bf16_weight == nullptr ||
      query_output == nullptr || key_input == nullptr ||
      key_bf16_weight == nullptr || key_output == nullptr ||
      device_base_position == nullptr || query_head_count == 0 ||
      key_head_count == 0 || batch == 0 || head_size == 0 ||
      head_size > 256 || query_input_stride < head_size ||
      key_input_stride < head_size || rotary_size == 0 ||
      rotary_size > head_size || (rotary_size & 1U) != 0 ||
      !std::isfinite(rope_theta) || rope_theta <= 1.0F ||
      query_head_count + key_head_count >
          std::numeric_limits<unsigned>::max() / batch) {
    return {ErrorCode::invalid_argument,
            "invalid indirect-position fused Q/K RMSNorm/RoPE arguments"};
  }
  hipLaunchKernelGGL(
      qk_rms_norm_rope,
      dim3(static_cast<unsigned>((query_head_count + key_head_count) * batch)),
      dim3(256), 0, stream, query_input, query_bf16_weight, query_output,
      query_head_count, query_input_stride, key_input, key_bf16_weight,
      key_output, key_head_count, key_input_stride, head_size, rotary_size,
      0U, device_base_position, rope_theta);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(
        error, "indirect-position fused Q/K RMSNorm/RoPE launch failed");
  }
  return Status::ok_status();
}

Status launch_kv_append(
    const __half* key,
    const float* value,
    __half* key_cache,
    __half* value_cache,
    std::size_t position,
    std::size_t capacity,
    hipStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr || value_cache == nullptr ||
      position >= capacity) {
    return {ErrorCode::invalid_argument, "invalid KV append arguments"};
  }
  constexpr unsigned elements = 4 * 256;
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(kv_append, dim3((elements + threads - 1U) / threads),
                     dim3(threads), 0, stream, key, value, key_cache, value_cache,
                     position, nullptr, capacity, 1U);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "KV append launch failed");
  }
  return Status::ok_status();
}

Status launch_kv_append_block(
    const __half* key,
    const float* value,
    __half* key_cache,
    __half* value_cache,
    std::size_t base_position,
    std::size_t capacity,
    unsigned batch,
    hipStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || batch == 0 || base_position >= capacity ||
      batch > capacity - base_position) {
    return {ErrorCode::invalid_argument, "invalid block KV append arguments"};
  }
  constexpr unsigned elements = 4 * 256;
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(
      kv_append, dim3((elements * batch + threads - 1U) / threads),
      dim3(threads), 0, stream, key, value, key_cache, value_cache,
      base_position, nullptr, capacity, batch);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "block KV append launch failed");
  }
  return Status::ok_status();
}

Status launch_kv_append_block_indirect_position(
    const __half* key,
    const float* value,
    __half* key_cache,
    __half* value_cache,
    const std::uint64_t* device_base_position,
    std::size_t capacity,
    unsigned batch,
    hipStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr ||
      value_cache == nullptr || device_base_position == nullptr || batch == 0) {
    return {ErrorCode::invalid_argument,
            "invalid indirect-position block KV append arguments"};
  }
  constexpr unsigned elements = 4 * 256;
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(
      kv_append, dim3((elements * batch + threads - 1U) / threads),
      dim3(threads), 0, stream, key, value, key_cache, value_cache, 0U,
      device_base_position, capacity, batch);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error,
                      "indirect-position block KV append launch failed");
  }
  return Status::ok_status();
}

Status launch_gated_attention_decode(
    const __half* query,
    const float* gate,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    std::size_t token_count,
    std::size_t capacity,
    hipStream_t stream) {
  if (query == nullptr || gate == nullptr || key_cache == nullptr || value_cache == nullptr ||
      output == nullptr || token_count == 0 || token_count > capacity) {
    return {ErrorCode::invalid_argument, "invalid gated attention decode arguments"};
  }
  hipLaunchKernelGGL(gated_attention_decode, dim3(24), dim3(256), 0, stream,
                     query, gate, key_cache, value_cache, output,
                     token_count, nullptr, capacity, 1U);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "gated attention decode launch failed");
  }
  return Status::ok_status();
}

Status launch_gated_attention_decode_block(
    const __half* query,
    const float* gate,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    std::size_t base_token_count,
    std::size_t capacity,
    unsigned batch,
    hipStream_t stream) {
  if (query == nullptr || gate == nullptr || key_cache == nullptr ||
      value_cache == nullptr || output == nullptr || batch == 0 ||
      base_token_count == 0 || base_token_count > capacity ||
      batch - 1U > capacity - base_token_count) {
    return {ErrorCode::invalid_argument, "invalid block gated attention decode arguments"};
  }
  hipLaunchKernelGGL(
      gated_attention_decode, dim3(24U * batch), dim3(256), 0, stream,
      query, gate, key_cache, value_cache, output,
      base_token_count, nullptr, capacity, batch);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "block gated attention decode launch failed");
  }
  return Status::ok_status();
}

Status launch_gated_attention_decode_block_indirect_position(
    const __half* query,
    const float* gate,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    const std::uint64_t* device_base_position,
    std::size_t capacity,
    unsigned batch,
    hipStream_t stream) {
  if (query == nullptr || gate == nullptr || key_cache == nullptr ||
      value_cache == nullptr || output == nullptr ||
      device_base_position == nullptr || batch == 0) {
    return {ErrorCode::invalid_argument,
            "invalid indirect-position block attention arguments"};
  }
  hipLaunchKernelGGL(
      gated_attention_decode, dim3(24U * batch), dim3(256), 0, stream,
      query, gate, key_cache, value_cache, output, 0U,
      device_base_position, capacity, batch);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(
        error, "indirect-position block attention launch failed");
  }
  return Status::ok_status();
}

Status launch_argmax(
    const float* logits,
    std::uint32_t* token,
    std::size_t vocabulary_size,
    hipStream_t stream) {
  if (logits == nullptr || token == nullptr || vocabulary_size == 0 ||
      vocabulary_size > std::numeric_limits<std::uint32_t>::max()) {
    return {ErrorCode::invalid_argument, "invalid argmax arguments"};
  }
  hipLaunchKernelGGL(argmax_logits, dim3(1), dim3(256), 0, stream,
                     logits, nullptr, token, vocabulary_size, vocabulary_size);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "argmax launch failed");
  }
  return Status::ok_status();
}

Status launch_argmax_batch(
    const float* logits,
    std::uint32_t* tokens,
    std::size_t vocabulary_size,
    std::size_t row_stride,
    unsigned batch,
    hipStream_t stream) {
  if (logits == nullptr || tokens == nullptr || vocabulary_size == 0 ||
      row_stride < vocabulary_size || batch == 0 || batch > 32 ||
      vocabulary_size > std::numeric_limits<std::uint32_t>::max()) {
    return {ErrorCode::invalid_argument, "invalid batched argmax arguments"};
  }
  hipLaunchKernelGGL(argmax_logits, dim3(batch), dim3(256), 0, stream,
                     logits, nullptr, tokens, vocabulary_size, row_stride);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "batched argmax launch failed");
  }
  return Status::ok_status();
}

Status launch_argmax_mapped(
    const float* logits,
    const std::uint32_t* token_ids,
    std::uint32_t* token,
    std::size_t vocabulary_size,
    hipStream_t stream) {
  if (logits == nullptr || token_ids == nullptr || token == nullptr ||
      vocabulary_size == 0 ||
      vocabulary_size > std::numeric_limits<std::uint32_t>::max()) {
    return {ErrorCode::invalid_argument, "invalid mapped argmax arguments"};
  }
  hipLaunchKernelGGL(argmax_logits, dim3(1), dim3(256), 0, stream,
                     logits, token_ids, token, vocabulary_size, vocabulary_size);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "mapped argmax launch failed");
  }
  return Status::ok_status();
}

Status launch_float_to_half(
    const float* input,
    __half* output,
    std::size_t elements,
    hipStream_t stream) {
  if (input == nullptr || output == nullptr || elements == 0) {
    return {ErrorCode::invalid_argument, "invalid float-to-half arguments"};
  }
  constexpr unsigned threads = 256;
  hipLaunchKernelGGL(float_to_half,
                     dim3(static_cast<unsigned>((elements + threads - 1U) / threads)),
                     dim3(threads), 0, stream, input, output, elements);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "float-to-half launch failed");
  }
  return Status::ok_status();
}

Status launch_concat_half(
    const __half* left,
    std::size_t left_elements,
    const __half* right,
    std::size_t right_elements,
    __half* output,
    hipStream_t stream) {
  if (left == nullptr || right == nullptr || output == nullptr ||
      left_elements == 0 || right_elements == 0 ||
      left_elements > std::numeric_limits<std::size_t>::max() - right_elements) {
    return {ErrorCode::invalid_argument, "invalid half concatenation arguments"};
  }
  constexpr unsigned threads = 256;
  const auto elements = left_elements + right_elements;
  hipLaunchKernelGGL(concat_half,
                     dim3(static_cast<unsigned>((elements + threads - 1U) / threads)),
                     dim3(threads), 0, stream, left, left_elements, right, right_elements, output);
  if (const auto error = hipGetLastError(); error != hipSuccess) {
    return hip_status(error, "half concatenation launch failed");
  }
  return Status::ok_status();
}

}  // namespace gfxinfer
