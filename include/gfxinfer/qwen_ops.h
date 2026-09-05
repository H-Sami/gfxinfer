// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include "gfxinfer/status.h"

namespace gfxinfer {

[[nodiscard]] Status launch_rms_norm(
    const __half* input,
    const std::uint16_t* bf16_weight,
    __half* output,
    std::size_t elements,
    float epsilon,
    bool add_unit_offset,
    hipStream_t stream);

[[nodiscard]] Status launch_rms_norm_batch(
    const __half* input,
    const std::uint16_t* bf16_weight,
    __half* output,
    std::size_t elements,
    unsigned batch,
    float epsilon,
    bool add_unit_offset,
    hipStream_t stream);

[[nodiscard]] Status launch_dual_rms_norm_concat(
    const __half* left,
    const std::uint16_t* left_bf16_weight,
    const __half* right,
    const std::uint16_t* right_bf16_weight,
    __half* concatenated_output,
    std::size_t elements,
    float epsilon,
    bool add_unit_offset,
    hipStream_t stream);

[[nodiscard]] Status launch_silu_multiply(
    const float* gate,
    const float* up,
    __half* output,
    std::size_t elements,
    hipStream_t stream);

// Produces the same A4G128 codes as FP16 SiLU-multiply followed by the regular
// activation quantizer, without materializing the intermediate FP16 matrix.
[[nodiscard]] Status launch_silu_multiply_quantize_i4(
    const float* gate,
    const float* up,
    std::uint8_t* packed_output,
    __half* scales,
    std::size_t elements_per_row,
    unsigned batch,
    hipStream_t stream);

[[nodiscard]] Status launch_residual_add(
    const __half* residual,
    const float* update,
    __half* output,
    std::size_t elements,
    hipStream_t stream);

// Adds the FP32 update to the FP16 residual, rounds the residual result to
// FP16, and RMS-normalizes that exact rounded value in one kernel.
[[nodiscard]] Status launch_residual_add_rms_norm_batch(
    const __half* residual,
    const float* update,
    __half* residual_output,
    const std::uint16_t* bf16_weight,
    __half* normalized_output,
    std::size_t elements,
    unsigned batch,
    float epsilon,
    bool add_unit_offset,
    hipStream_t stream);

[[nodiscard]] Status launch_gdn_conv_update(
    const float* projected_qkv,
    const std::uint16_t* bf16_conv_weight,
    __half* conv_state,
    __half* convolved_qkv,
    hipStream_t stream);

[[nodiscard]] Status launch_gdn_conv_update_block(
    const float* projected_qkv,
    const std::uint16_t* bf16_conv_weight,
    const __half* initial_conv_state,
    __half* final_conv_state,
    __half* convolved_qkv,
    __half* appended_log,
    std::size_t channels,
    unsigned batch,
    hipStream_t stream);

[[nodiscard]] Status launch_gdn_recurrent_decode(
    const __half* convolved_qkv,
    const float* projected_z,
    const float* projected_a,
    const float* projected_b,
    const std::uint16_t* bf16_a_log,
    const std::uint16_t* bf16_dt_bias,
    const std::uint16_t* bf16_norm_weight,
    float* recurrent_state,
    __half* output,
    hipStream_t stream);

[[nodiscard]] Status launch_gdn_recurrent_decode_block(
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
    hipStream_t stream);

[[nodiscard]] Status launch_gdn_restore_prefix(
    __half* conv_state,
    float* recurrent_state,
    const __half* appended_log,
    const float* key_log,
    const float* delta_log,
    const float* decay_log,
    std::size_t channels,
    unsigned accepted_tokens,
    hipStream_t stream);

[[nodiscard]] Status launch_gdn_restore_prefix_all(
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
    hipStream_t stream);

[[nodiscard]] Status launch_head_rms_norm_rope(
    const float* input,
    const std::uint16_t* bf16_weight,
    __half* output,
    std::size_t head_count,
    std::size_t head_size,
    std::size_t input_stride,
    std::size_t rotary_size,
    std::uint64_t position,
    float rope_theta,
    hipStream_t stream);

[[nodiscard]] Status launch_head_rms_norm_rope_block(
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
    hipStream_t stream);

[[nodiscard]] Status launch_qk_rms_norm_rope_block(
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
    hipStream_t stream);

// Graph-capture variant: all attention layers read the round's base position
// from one stable device scalar, avoiding per-node graph parameter updates.
[[nodiscard]] Status launch_qk_rms_norm_rope_block_indirect_position(
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
    hipStream_t stream);

[[nodiscard]] Status launch_kv_append(
    const __half* key,
    const float* value,
    __half* key_cache,
    __half* value_cache,
    std::size_t position,
    std::size_t capacity,
    hipStream_t stream);

[[nodiscard]] Status launch_kv_append_block(
    const __half* key,
    const float* value,
    __half* key_cache,
    __half* value_cache,
    std::size_t base_position,
    std::size_t capacity,
    unsigned batch,
    hipStream_t stream);

[[nodiscard]] Status launch_kv_append_block_indirect_position(
    const __half* key,
    const float* value,
    __half* key_cache,
    __half* value_cache,
    const std::uint64_t* device_base_position,
    std::size_t capacity,
    unsigned batch,
    hipStream_t stream);

[[nodiscard]] Status launch_gated_attention_decode(
    const __half* query,
    const float* gate,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    std::size_t token_count,
    std::size_t capacity,
    hipStream_t stream);

[[nodiscard]] Status launch_gated_attention_decode_block(
    const __half* query,
    const float* gate,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    std::size_t base_token_count,
    std::size_t capacity,
    unsigned batch,
    hipStream_t stream);

[[nodiscard]] Status launch_gated_attention_decode_block_indirect_position(
    const __half* query,
    const float* gate,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    const std::uint64_t* device_base_position,
    std::size_t capacity,
    unsigned batch,
    hipStream_t stream);

[[nodiscard]] Status launch_argmax(
    const float* logits,
    std::uint32_t* token,
    std::size_t vocabulary_size,
    hipStream_t stream);

[[nodiscard]] Status launch_argmax_batch(
    const float* logits,
    std::uint32_t* tokens,
    std::size_t vocabulary_size,
    std::size_t row_stride,
    unsigned batch,
    hipStream_t stream);

[[nodiscard]] Status launch_argmax_mapped(
    const float* logits,
    const std::uint32_t* token_ids,
    std::uint32_t* token,
    std::size_t vocabulary_size,
    hipStream_t stream);

[[nodiscard]] Status launch_float_to_half(
    const float* input,
    __half* output,
    std::size_t elements,
    hipStream_t stream);

[[nodiscard]] Status launch_concat_half(
    const __half* left,
    std::size_t left_elements,
    const __half* right,
    std::size_t right_elements,
    __half* output,
    hipStream_t stream);

}  // namespace gfxinfer
