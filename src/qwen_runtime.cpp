// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/qwen_runtime.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gfxinfer/device.h"
#include "gfxinfer/lowbit_gemv.h"
#include "gfxinfer/qwen38_spec.h"
#include "gfxinfer/qwen_ops.h"

namespace gfxinfer {
namespace {

constexpr std::size_t kGdnLayerCount = 48;
constexpr std::size_t kAttentionLayerCount = 16;
constexpr std::size_t kHidden = Qwen38Spec::hidden_size;
constexpr std::size_t kIntermediate = Qwen38Spec::intermediate_size;
constexpr std::size_t kGdnQkv = 10240;
constexpr std::size_t kGdnKey = 2048;
constexpr std::size_t kGdnValue = 6144;
constexpr std::size_t kGdnHeads = 48;
constexpr std::size_t kGdnHeadSize = 128;
constexpr std::size_t kGdnStateElements = kGdnHeads * kGdnHeadSize * kGdnHeadSize;
constexpr std::size_t kConvStateElements = kGdnQkv * 4;
constexpr std::size_t kQueryElements = 24 * 256;
constexpr std::size_t kQueryProjectionElements = kQueryElements * 2;
constexpr std::size_t kKvElements = 4 * 256;
constexpr unsigned kMaximumBlock = 32;
// The MTP head is only a proposal model: a miss is corrected by the full
// target verifier.  Searching the high-frequency prefix avoids reading the
// cold half of the 248K-row head on every draft while preserving target-model
// correctness.  Qwen's ordinary text vocabulary is ordered densely enough
// that this prefix covers the overwhelming majority of generated tokens.
constexpr std::size_t kMtpDraftVocabulary = 98304;

struct LayerBindings {
  bool attention{false};
  std::size_t state_index{0};
  const DeviceTensorView* input_norm{nullptr};
  const DeviceTensorView* post_norm{nullptr};
  const DeviceTensorView* mlp_gate{nullptr};
  const DeviceTensorView* mlp_up{nullptr};
  const DeviceTensorView* mlp_down{nullptr};
  const DeviceTensorView* qkv{nullptr};
  const DeviceTensorView* z{nullptr};
  const DeviceTensorView* a{nullptr};
  const DeviceTensorView* b{nullptr};
  const DeviceTensorView* convolution{nullptr};
  const DeviceTensorView* a_log{nullptr};
  const DeviceTensorView* dt_bias{nullptr};
  const DeviceTensorView* gdn_norm{nullptr};
  const DeviceTensorView* q{nullptr};
  const DeviceTensorView* k{nullptr};
  const DeviceTensorView* v{nullptr};
  const DeviceTensorView* q_norm{nullptr};
  const DeviceTensorView* k_norm{nullptr};
  const DeviceTensorView* out{nullptr};
};

struct CapturedGraph {
  hipGraph_t graph{nullptr};
  hipGraphExec_t executable{nullptr};
};

[[nodiscard]] Status launch_matrix(
    const DeviceTensorView* tensor,
    const __half* input,
    float* output,
    std::uint8_t* activation_codes,
    __half* activation_scales,
    bool tiled_weights,
    bool sequential_w3,
    ActivationMode activation_mode,
    unsigned batch,
    bool prepare_activation,
    hipStream_t stream) {
  if (tensor == nullptr || tensor->rank != 2 || tensor->data == nullptr ||
      tensor->auxiliary == nullptr) {
    return {ErrorCode::invalid_argument, "runtime matrix binding is invalid"};
  }
  const auto rows = static_cast<std::size_t>(tensor->dimensions[0]);
  const auto columns = static_cast<std::size_t>(tensor->dimensions[1]);
  if (!tiled_weights) {
    if (batch > 1) {
      return launch_lowbit_gemv_batch(
          tensor->quantization,
          reinterpret_cast<const std::uint8_t*>(tensor->data),
          reinterpret_cast<const __half*>(tensor->auxiliary), input, output,
          rows, columns, batch, stream);
    }
    return launch_lowbit_gemv(
        tensor->quantization,
        reinterpret_cast<const std::uint8_t*>(tensor->data),
        reinterpret_cast<const __half*>(tensor->auxiliary), input, output,
        rows, columns, stream);
  }
  const bool use_int4 =
      activation_mode == ActivationMode::int4 ||
      (activation_mode == ActivationMode::int4_w2 &&
       tensor->quantization == Quantization::w2g128) ||
      (activation_mode == ActivationMode::int4_w4 &&
       tensor->quantization == Quantization::w4g128);
  if (use_int4) {
    if (prepare_activation) {
      auto status = launch_quantize_activation_i4(
          input, activation_codes, activation_scales, columns, batch, stream);
      if (!status.ok()) return status;
    }
    return launch_gfx12_i4_gemm(
        tensor->quantization,
        reinterpret_cast<const std::uint8_t*>(tensor->data),
        reinterpret_cast<const __half*>(tensor->auxiliary),
        activation_codes, activation_scales, output,
        rows, columns, batch, sequential_w3, stream);
  }
  if (activation_mode == ActivationMode::int8) {
    if (prepare_activation) {
      auto status = launch_quantize_activation_i8(
          input, reinterpret_cast<std::int8_t*>(activation_codes),
          activation_scales, columns, batch, stream);
      if (!status.ok()) return status;
    }
    return launch_gfx12_i8_gemm(
        tensor->quantization,
        reinterpret_cast<const std::uint8_t*>(tensor->data),
        reinterpret_cast<const __half*>(tensor->auxiliary),
        reinterpret_cast<const std::int8_t*>(activation_codes),
        activation_scales, output, rows, columns, batch, stream);
  }
  return launch_gfx12_f16_gemm(
      tensor->quantization,
      reinterpret_cast<const std::uint8_t*>(tensor->data),
      reinterpret_cast<const __half*>(tensor->auxiliary),
      input, output, rows, columns, batch, stream);
}

}  // namespace

struct QwenDecodeSession::Impl {
  ~Impl() {
    if (engine != nullptr) (void)hipSetDevice(engine->load_summary().device.index);
    for (auto& graph : speculative_graphs) {
      if (graph.executable != nullptr) {
        (void)hipGraphExecDestroy(graph.executable);
      }
      if (graph.graph != nullptr) (void)hipGraphDestroy(graph.graph);
    }
    if (arena != nullptr) (void)hipFree(arena);
  }

  Engine* engine{nullptr};
  DecodeSessionConfig config;
  DecodeSessionSummary summary;
  std::size_t current_position{0};
  std::size_t mtp_position{0};
  bool mtp_chain_active{false};
  bool record_rollback{false};
  bool tiled_weights{false};
  bool sequential_w3{false};
  bool gdn_state_parity{false};
  std::byte* arena{nullptr};
  std::vector<LayerBindings> layers;
  std::array<CapturedGraph, 2> speculative_graphs;
  LayerBindings mtp_layer;
  const DeviceTensorView* embedding{nullptr};
  const DeviceTensorView* final_norm{nullptr};
  const DeviceTensorView* lm_head{nullptr};
  DeviceTensorView mtp_draft_head;
  const DeviceTensorView* mtp_draft_token_ids{nullptr};
  const DeviceTensorView* mtp_fc{nullptr};
  const DeviceTensorView* mtp_pre_embedding_norm{nullptr};
  const DeviceTensorView* mtp_pre_hidden_norm{nullptr};
  const DeviceTensorView* mtp_final_norm{nullptr};
  const __half* last_hidden{nullptr};
  __half* mtp_hidden{nullptr};
  __half* target_hidden_snapshot{nullptr};

  __half* hidden_a{nullptr};
  __half* hidden_b{nullptr};
  __half* normalized{nullptr};
  __half* convolved_qkv{nullptr};
  __half* gdn_output{nullptr};
  __half* query{nullptr};
  __half* key{nullptr};
  __half* attention_output{nullptr};
  __half* post_attention{nullptr};
  __half* post_normalized{nullptr};
  __half* activated{nullptr};
  float* projected_main{nullptr};
  float* projected_z{nullptr};
  float* projected_a{nullptr};
  float* projected_b{nullptr};
  float* projected_k{nullptr};
  float* projected_v{nullptr};
  float* mixer_update{nullptr};
  float* gate_output{nullptr};
  float* up_output{nullptr};
  float* mlp_update{nullptr};
  float* logits{nullptr};
  std::uint32_t* selected_token{nullptr};
  std::uint32_t* draft_tokens{nullptr};
  std::uint32_t* graph_seed_token{nullptr};
  std::uint64_t* graph_base_position{nullptr};
  std::uint64_t* graph_mtp_positions{nullptr};
  std::uint32_t graph_host_seed_token{0};
  std::uint64_t graph_host_base_position{0};
  std::array<std::uint64_t, kMaximumBlock> graph_host_mtp_positions{};
  __half* conv_states{nullptr};
  float* gdn_states{nullptr};
  __half* conv_snapshot{nullptr};
  float* gdn_snapshot{nullptr};
  __half* conv_rollback_log{nullptr};
  float* gdn_key_log{nullptr};
  float* gdn_delta_log{nullptr};
  float* gdn_decay_log{nullptr};
  __half* key_caches{nullptr};
  __half* value_caches{nullptr};
  __half* mtp_key_cache{nullptr};
  __half* mtp_value_cache{nullptr};
  std::uint8_t* activation_codes{nullptr};
  __half* activation_scales{nullptr};

  [[nodiscard]] Status snapshot_target_state() {
    const auto stream = engine->compute_stream();
    if (last_hidden != nullptr) {
      if (const auto error = hipMemcpyAsync(
              target_hidden_snapshot, last_hidden, kHidden * sizeof(__half),
              hipMemcpyDeviceToDevice, stream);
          error != hipSuccess) {
        return hip_status(error, "target hidden-state snapshot failed");
      }
    }
    return Status::ok_status();
  }

  [[nodiscard]] Status restore_target_state() {
    last_hidden = target_hidden_snapshot;
    return Status::ok_status();
  }

  [[nodiscard]] Status restore_logged_prefix(unsigned accepted_tokens) {
    if (accepted_tokens == 0) return Status::ok_status();
    return launch_gdn_restore_prefix_all(
        conv_states, gdn_states, conv_rollback_log, gdn_key_log,
        gdn_delta_log, gdn_decay_log, kGdnQkv,
        static_cast<unsigned>(kGdnLayerCount), kMaximumBlock,
        accepted_tokens, engine->compute_stream());
  }

  [[nodiscard]] Status matrix(
      const DeviceTensorView* tensor,
      const __half* input,
      float* output,
      unsigned batch = 1,
      bool prepare_activation = true) {
    return launch_matrix(tensor, input, output, activation_codes,
                         activation_scales, tiled_weights,
                         sequential_w3,
                         config.activation_mode,
                         batch, prepare_activation, engine->compute_stream());
  }

  [[nodiscard]] bool uses_i4(const DeviceTensorView* tensor) const noexcept {
    return tensor != nullptr &&
           (config.activation_mode == ActivationMode::int4 ||
            (config.activation_mode == ActivationMode::int4_w2 &&
             tensor->quantization == Quantization::w2g128) ||
            (config.activation_mode == ActivationMode::int4_w4 &&
             tensor->quantization == Quantization::w4g128));
  }

  [[nodiscard]] Status matrices(
      std::span<const DeviceTensorView* const> tensors,
      std::span<float* const> outputs,
      const __half* input,
      unsigned batch = 1) {
    if (tensors.size() != outputs.size() || tensors.empty()) {
      return {ErrorCode::invalid_argument, "fused matrix list is invalid"};
    }
    const auto format = tensors.front()->quantization;
    const auto columns = tensors.front()->dimensions[1];
    const bool compatible = std::all_of(
        tensors.begin(), tensors.end(), [&](const DeviceTensorView* tensor) {
          return tensor != nullptr && tensor->rank == 2 &&
                 tensor->quantization == format && tensor->dimensions[1] == columns;
        });
    const bool use_i4 = uses_i4(tensors.front());
    if (tiled_weights && compatible && use_i4 && tensors.size() >= 2 &&
        tensors.size() <= 4) {
      auto status = launch_quantize_activation_i4(
          input, activation_codes, activation_scales,
          static_cast<std::size_t>(columns), batch, engine->compute_stream());
      if (!status.ok()) return status;
      std::array<Gfx12I4FusedMatrix, 4> bindings{};
      for (std::size_t index = 0; index < tensors.size(); ++index) {
        bindings[index] = {
            reinterpret_cast<const std::uint8_t*>(tensors[index]->data),
            reinterpret_cast<const __half*>(tensors[index]->auxiliary),
            outputs[index],
            static_cast<std::size_t>(tensors[index]->dimensions[0]),
        };
      }
      return launch_gfx12_i4_gemm_fused(
          format,
          std::span<const Gfx12I4FusedMatrix>(bindings.data(), tensors.size()),
          activation_codes, activation_scales,
          static_cast<std::size_t>(columns), batch, sequential_w3,
          engine->compute_stream());
    }
    for (std::size_t index = 0; index < tensors.size(); ++index) {
      auto status = matrix(tensors[index], input, outputs[index], batch,
                           index == 0 || !compatible);
      if (!status.ok()) return status;
    }
    return Status::ok_status();
  }

  [[nodiscard]] Status launch_mlp_down(
      const DeviceTensorView* down,
      unsigned batch = 1) {
    const auto stream = engine->compute_stream();
    if (uses_i4(down)) {
      auto status = launch_silu_multiply_quantize_i4(
          gate_output, up_output, activation_codes, activation_scales,
          kIntermediate, batch, stream);
      if (!status.ok()) return status;
      return matrix(down, activated, mlp_update, batch, false);
    }
    auto status = launch_silu_multiply(
        gate_output, up_output, activated,
        static_cast<std::size_t>(batch) * kIntermediate, stream);
    if (!status.ok()) return status;
    return matrix(down, activated, mlp_update, batch);
  }

  [[nodiscard]] Status launch_layer(
      const LayerBindings& layer,
      const __half* input,
      __half* output,
      const std::uint16_t* next_norm_weight) {
    const auto stream = engine->compute_stream();
    Status status;
    if (layer.attention) {
      const std::array<const DeviceTensorView*, 3> weights{layer.q, layer.k, layer.v};
      const std::array<float*, 3> outputs{projected_main, projected_k, projected_v};
      status = matrices(weights, outputs, normalized);
      if (!status.ok()) return status;
      status = launch_qk_rms_norm_rope_block(
          projected_main,
          reinterpret_cast<const std::uint16_t*>(layer.q_norm->data),
          query, 24, 512, projected_k,
          reinterpret_cast<const std::uint16_t*>(layer.k_norm->data),
          key, 4, 256, 256, Qwen38Spec::rotary_size, current_position, 1,
          Qwen38Spec::rope_theta, stream);
      if (!status.ok()) return status;
      const auto cache_offset = layer.state_index * config.maximum_context_tokens * kKvElements;
      auto* key_cache = key_caches + cache_offset;
      auto* value_cache = value_caches + cache_offset;
      status = launch_kv_append(key, projected_v, key_cache, value_cache,
                                current_position, config.maximum_context_tokens, stream);
      if (!status.ok()) return status;
      status = launch_gated_attention_decode(
          query, projected_main, key_cache, value_cache, attention_output,
          current_position + 1, config.maximum_context_tokens, stream);
      if (!status.ok()) return status;
      status = matrix(layer.out, attention_output, mixer_update);
    } else {
      const std::array<const DeviceTensorView*, 4> weights{
          layer.qkv, layer.z, layer.a, layer.b};
      const std::array<float*, 4> outputs{
          projected_main, projected_z, projected_a, projected_b};
      status = matrices(weights, outputs, normalized);
      if (!status.ok()) return status;
      auto* conv_state = conv_states + layer.state_index * kConvStateElements;
      auto* recurrent_state = gdn_states + layer.state_index * kGdnStateElements;
      status = launch_gdn_conv_update(
          projected_main,
          reinterpret_cast<const std::uint16_t*>(layer.convolution->data),
          conv_state, convolved_qkv, stream);
      if (!status.ok()) return status;
      status = launch_gdn_recurrent_decode(
          convolved_qkv, projected_z, projected_a, projected_b,
          reinterpret_cast<const std::uint16_t*>(layer.a_log->data),
          reinterpret_cast<const std::uint16_t*>(layer.dt_bias->data),
          reinterpret_cast<const std::uint16_t*>(layer.gdn_norm->data),
          recurrent_state, gdn_output, stream);
      if (!status.ok()) return status;
      status = matrix(layer.out, gdn_output, mixer_update);
    }
    if (!status.ok()) return status;
    status = launch_residual_add_rms_norm_batch(
        input, mixer_update, post_attention,
        reinterpret_cast<const std::uint16_t*>(layer.post_norm->data),
        post_normalized, kHidden, 1, Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!status.ok()) return status;
    const std::array<const DeviceTensorView*, 2> mlp_weights{
        layer.mlp_gate, layer.mlp_up};
    const std::array<float*, 2> mlp_outputs{gate_output, up_output};
    status = matrices(mlp_weights, mlp_outputs, post_normalized);
    if (!status.ok()) return status;
    status = launch_mlp_down(layer.mlp_down);
    if (!status.ok()) return status;
    return launch_residual_add_rms_norm_batch(
        post_attention, mlp_update, output, next_norm_weight, normalized,
        kHidden, 1, Qwen38Spec::rms_norm_epsilon, true, stream);
  }

  [[nodiscard]] Status launch_layer_block(
      const LayerBindings& layer,
      const __half* input,
      __half* output,
      unsigned batch,
      const std::uint16_t* next_norm_weight,
      const std::uint64_t* device_base_position = nullptr) {
    const auto stream = engine->compute_stream();
    Status status;
    if (layer.attention) {
      const std::array<const DeviceTensorView*, 3> weights{layer.q, layer.k, layer.v};
      const std::array<float*, 3> outputs{projected_main, projected_k, projected_v};
      status = matrices(weights, outputs, normalized, batch);
      if (!status.ok()) return status;
      const auto cache_offset = layer.state_index * config.maximum_context_tokens * kKvElements;
      auto* key_cache = key_caches + cache_offset;
      auto* value_cache = value_caches + cache_offset;
      status = device_base_position == nullptr
                   ? launch_qk_rms_norm_rope_block(
                         projected_main,
                         reinterpret_cast<const std::uint16_t*>(
                             layer.q_norm->data),
                         query, 24, 512, projected_k,
                         reinterpret_cast<const std::uint16_t*>(
                             layer.k_norm->data),
                         key, 4, 256, 256, Qwen38Spec::rotary_size,
                         current_position, batch, Qwen38Spec::rope_theta,
                         stream)
                   : launch_qk_rms_norm_rope_block_indirect_position(
                         projected_main,
                         reinterpret_cast<const std::uint16_t*>(
                             layer.q_norm->data),
                         query, 24, 512, projected_k,
                         reinterpret_cast<const std::uint16_t*>(
                             layer.k_norm->data),
                         key, 4, 256, 256, Qwen38Spec::rotary_size,
                         device_base_position, batch, Qwen38Spec::rope_theta,
                         stream);
      if (!status.ok()) return status;
      status = device_base_position == nullptr
                   ? launch_kv_append_block(
                         key, projected_v, key_cache, value_cache,
                         current_position, config.maximum_context_tokens,
                         batch, stream)
                   : launch_kv_append_block_indirect_position(
                         key, projected_v, key_cache, value_cache,
                         device_base_position, config.maximum_context_tokens,
                         batch, stream);
      if (!status.ok()) return status;
      status = device_base_position == nullptr
                   ? launch_gated_attention_decode_block(
                         query, projected_main, key_cache, value_cache,
                         attention_output, current_position + 1,
                         config.maximum_context_tokens, batch, stream)
                   : launch_gated_attention_decode_block_indirect_position(
                         query, projected_main, key_cache, value_cache,
                         attention_output, device_base_position,
                         config.maximum_context_tokens, batch, stream);
      if (!status.ok()) return status;
      status = matrix(layer.out, attention_output, mixer_update, batch);
    } else {
      const std::array<const DeviceTensorView*, 4> weights{
          layer.qkv, layer.z, layer.a, layer.b};
      const std::array<float*, 4> outputs{
          projected_main, projected_z, projected_a, projected_b};
      status = matrices(weights, outputs, normalized, batch);
      if (!status.ok()) return status;
      auto* conv_state = conv_states + layer.state_index * kConvStateElements;
      auto* recurrent_state = gdn_states + layer.state_index * kGdnStateElements;
      auto* final_conv_state = record_rollback
                                   ? conv_snapshot + layer.state_index * kConvStateElements
                                   : conv_state;
      auto* final_recurrent_state = record_rollback
                                        ? gdn_snapshot + layer.state_index * kGdnStateElements
                                        : recurrent_state;
      auto* conv_log = record_rollback
                           ? conv_rollback_log +
                                 layer.state_index * kMaximumBlock * kGdnQkv
                           : nullptr;
      auto* key_log = record_rollback
                          ? gdn_key_log +
                                layer.state_index * kMaximumBlock * kGdnKey
                          : nullptr;
      auto* delta_log = record_rollback
                            ? gdn_delta_log +
                                  layer.state_index * kMaximumBlock * kGdnValue
                            : nullptr;
      auto* decay_log = record_rollback
                            ? gdn_decay_log +
                                  layer.state_index * kMaximumBlock * kGdnHeads
                            : nullptr;
      status = launch_gdn_conv_update_block(
          projected_main,
          reinterpret_cast<const std::uint16_t*>(layer.convolution->data),
          conv_state, final_conv_state, convolved_qkv, conv_log,
          kGdnQkv, batch, stream);
      if (!status.ok()) return status;
      status = launch_gdn_recurrent_decode_block(
          convolved_qkv, projected_z, projected_a, projected_b,
          reinterpret_cast<const std::uint16_t*>(layer.a_log->data),
          reinterpret_cast<const std::uint16_t*>(layer.dt_bias->data),
          reinterpret_cast<const std::uint16_t*>(layer.gdn_norm->data),
          recurrent_state, final_recurrent_state, gdn_output,
          key_log, delta_log, decay_log, batch, stream);
      if (!status.ok()) return status;
      status = matrix(layer.out, gdn_output, mixer_update, batch);
    }
    if (!status.ok()) return status;
    status = launch_residual_add_rms_norm_batch(
        input, mixer_update, post_attention,
        reinterpret_cast<const std::uint16_t*>(layer.post_norm->data),
        post_normalized, kHidden, batch, Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!status.ok()) return status;
    const std::array<const DeviceTensorView*, 2> mlp_weights{
        layer.mlp_gate, layer.mlp_up};
    const std::array<float*, 2> mlp_outputs{gate_output, up_output};
    status = matrices(mlp_weights, mlp_outputs, post_normalized, batch);
    if (!status.ok()) return status;
    status = launch_mlp_down(layer.mlp_down, batch);
    if (!status.ok()) return status;
    return launch_residual_add_rms_norm_batch(
        post_attention, mlp_update, output, next_norm_weight, normalized,
        kHidden, batch, Qwen38Spec::rms_norm_epsilon, true, stream);
  }

  [[nodiscard]] Status enqueue_verifier_body(
      std::uint32_t direct_seed_token,
      const std::uint32_t* indirect_seed_token,
      const std::uint32_t* device_drafts,
      unsigned draft_count,
      const std::uint64_t* device_base_position) {
    const auto stream = engine->compute_stream();
    auto status = indirect_seed_token != nullptr
                      ? (tiled_weights
                             ? launch_lowbit_row_gather_tiled_indirect(
                                   embedding->quantization,
                                   reinterpret_cast<const std::uint8_t*>(
                                       embedding->data),
                                   reinterpret_cast<const __half*>(
                                       embedding->auxiliary),
                                   hidden_a, indirect_seed_token,
                                   static_cast<std::size_t>(
                                       embedding->dimensions[0]),
                                   kHidden, stream)
                             : launch_lowbit_row_gather_indirect(
                                   embedding->quantization,
                                   reinterpret_cast<const std::uint8_t*>(
                                       embedding->data),
                                   reinterpret_cast<const __half*>(
                                       embedding->auxiliary),
                                   hidden_a, indirect_seed_token,
                                   static_cast<std::size_t>(
                                       embedding->dimensions[0]),
                                   kHidden, stream))
                      : (tiled_weights
                             ? launch_lowbit_row_gather_tiled(
                                   embedding->quantization,
                                   reinterpret_cast<const std::uint8_t*>(
                                       embedding->data),
                                   reinterpret_cast<const __half*>(
                                       embedding->auxiliary),
                                   hidden_a, direct_seed_token,
                                   static_cast<std::size_t>(
                                       embedding->dimensions[0]),
                                   kHidden, stream)
                             : launch_lowbit_row_gather(
                                   embedding->quantization,
                                   reinterpret_cast<const std::uint8_t*>(
                                       embedding->data),
                                   reinterpret_cast<const __half*>(
                                       embedding->auxiliary),
                                   hidden_a, direct_seed_token,
                                   static_cast<std::size_t>(
                                       embedding->dimensions[0]),
                                   kHidden, stream));
    if (!status.ok()) return status;
    for (unsigned index = 0; index < draft_count; ++index) {
      auto* destination = hidden_a + static_cast<std::size_t>(index + 1) * kHidden;
      status = tiled_weights
                   ? launch_lowbit_row_gather_tiled_indirect(
                         embedding->quantization,
                         reinterpret_cast<const std::uint8_t*>(embedding->data),
                         reinterpret_cast<const __half*>(embedding->auxiliary),
                         destination, device_drafts + index,
                         static_cast<std::size_t>(embedding->dimensions[0]),
                         kHidden, stream)
                   : launch_lowbit_row_gather_indirect(
                         embedding->quantization,
                         reinterpret_cast<const std::uint8_t*>(embedding->data),
                         reinterpret_cast<const __half*>(embedding->auxiliary),
                         destination, device_drafts + index,
                         static_cast<std::size_t>(embedding->dimensions[0]),
                         kHidden, stream);
      if (!status.ok()) return status;
    }

    const unsigned batch = draft_count + 1;
    __half* current = hidden_a;
    __half* next = hidden_b;
    status = launch_rms_norm_batch(
        current,
        reinterpret_cast<const std::uint16_t*>(layers.front().input_norm->data),
        normalized, kHidden, batch, Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!status.ok()) return status;
    for (std::size_t index = 0; index < layers.size(); ++index) {
      const auto* next_norm = reinterpret_cast<const std::uint16_t*>(
          index + 1 < layers.size() ? layers[index + 1].input_norm->data
                                    : final_norm->data);
      status = launch_layer_block(
          layers[index], current, next, batch, next_norm,
          device_base_position);
      if (!status.ok()) return status;
      std::swap(current, next);
    }
    status = matrix(lm_head, normalized, logits, batch);
    if (!status.ok()) return status;
    return launch_argmax_batch(
        logits, selected_token, Qwen38Spec::tokenizer_vocabulary_size,
        Qwen38Spec::vocabulary_size, batch, stream);
  }

  [[nodiscard]] Status prepare_speculative_graphs() {
    const unsigned batch = config.graph_verifier_batch;
    if (batch == 0) return Status::ok_status();
    const unsigned draft_count = batch - 1U;
    record_rollback = true;
    for (unsigned parity = 0; parity < 2; ++parity) {
      if (parity != 0) {
        std::swap(conv_states, conv_snapshot);
        std::swap(gdn_states, gdn_snapshot);
      }
      const auto stream = engine->compute_stream();
      if (const auto error = hipStreamBeginCapture(
              stream, hipStreamCaptureModeThreadLocal);
          error != hipSuccess) {
        if (parity != 0) {
          std::swap(conv_states, conv_snapshot);
          std::swap(gdn_states, gdn_snapshot);
        }
        record_rollback = false;
        return hip_status(error, "speculative graph capture start failed");
      }
      Status status = Status::ok_status();
      for (unsigned index = 0; index < draft_count; ++index) {
        const auto* indirect_token =
            index == 0 ? graph_seed_token : draft_tokens + index - 1U;
        const auto* conditioning_hidden =
            index == 0 ? target_hidden_snapshot : mtp_hidden;
        status = launch_mtp_proposal(
            0U, indirect_token, conditioning_hidden, 0U,
            draft_tokens + index, graph_mtp_positions + index);
        if (!status.ok()) break;
      }
      if (status.ok()) {
        status = enqueue_verifier_body(
            0U, graph_seed_token, draft_tokens, draft_count,
            graph_base_position);
      }
      hipGraph_t graph = nullptr;
      const auto end_error = hipStreamEndCapture(stream, &graph);
      if (!status.ok() || end_error != hipSuccess) {
        if (graph != nullptr) (void)hipGraphDestroy(graph);
        if (parity != 0) {
          std::swap(conv_states, conv_snapshot);
          std::swap(gdn_states, gdn_snapshot);
        }
        record_rollback = false;
        return !status.ok()
                   ? status
                   : hip_status(end_error,
                                "speculative graph capture end failed");
      }
      hipGraphExec_t executable = nullptr;
      if (const auto error = hipGraphInstantiate(
              &executable, graph, nullptr, nullptr, 0);
          error != hipSuccess) {
        (void)hipGraphDestroy(graph);
        if (parity != 0) {
          std::swap(conv_states, conv_snapshot);
          std::swap(gdn_states, gdn_snapshot);
        }
        record_rollback = false;
        return hip_status(error, "speculative graph instantiation failed");
      }
      speculative_graphs[parity] = {graph, executable};
      if (parity != 0) {
        std::swap(conv_states, conv_snapshot);
        std::swap(gdn_states, gdn_snapshot);
      }
    }
    record_rollback = false;
    return Status::ok_status();
  }

  [[nodiscard]] Status launch_mtp_proposal(
      std::uint32_t direct_token,
      const std::uint32_t* indirect_token,
      const __half* conditioning_hidden,
      std::size_t position,
      std::uint32_t* output_token,
      const std::uint64_t* device_position = nullptr) {
    if (conditioning_hidden == nullptr) {
      return {ErrorCode::invalid_argument, "invalid MTP proposal buffers"};
    }
    const auto stream = engine->compute_stream();
    auto status = indirect_token != nullptr
                      ? (tiled_weights
                             ? launch_lowbit_row_gather_tiled_indirect(
                                   embedding->quantization,
                                   reinterpret_cast<const std::uint8_t*>(embedding->data),
                                   reinterpret_cast<const __half*>(embedding->auxiliary),
                                   hidden_b, indirect_token,
                                   static_cast<std::size_t>(embedding->dimensions[0]),
                                   kHidden, stream)
                             : launch_lowbit_row_gather_indirect(
                                   embedding->quantization,
                                   reinterpret_cast<const std::uint8_t*>(embedding->data),
                                   reinterpret_cast<const __half*>(embedding->auxiliary),
                                   hidden_b, indirect_token,
                                   static_cast<std::size_t>(embedding->dimensions[0]),
                                   kHidden, stream))
                      : (tiled_weights
                             ? launch_lowbit_row_gather_tiled(
                                   embedding->quantization,
                                   reinterpret_cast<const std::uint8_t*>(embedding->data),
                                   reinterpret_cast<const __half*>(embedding->auxiliary),
                                   hidden_b, direct_token,
                                   static_cast<std::size_t>(embedding->dimensions[0]),
                                   kHidden, stream)
                             : launch_lowbit_row_gather(
                                   embedding->quantization,
                                   reinterpret_cast<const std::uint8_t*>(embedding->data),
                                   reinterpret_cast<const __half*>(embedding->auxiliary),
                                   hidden_b, direct_token,
                                   static_cast<std::size_t>(embedding->dimensions[0]),
                                   kHidden, stream));
    if (!status.ok()) return status;
    status = launch_dual_rms_norm_concat(
        hidden_b,
        reinterpret_cast<const std::uint16_t*>(mtp_pre_embedding_norm->data),
        conditioning_hidden,
        reinterpret_cast<const std::uint16_t*>(mtp_pre_hidden_norm->data),
        convolved_qkv, kHidden, Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!status.ok()) return status;
    if (tiled_weights && uses_i4(mtp_fc)) {
      status = launch_quantize_activation_i4(
          convolved_qkv, activation_codes, activation_scales,
          static_cast<std::size_t>(mtp_fc->dimensions[1]), 1, stream);
      if (!status.ok()) return status;
      status = launch_gfx12_i4_gemm_half(
          mtp_fc->quantization,
          reinterpret_cast<const std::uint8_t*>(mtp_fc->data),
          reinterpret_cast<const __half*>(mtp_fc->auxiliary),
          activation_codes, activation_scales, gdn_output,
          static_cast<std::size_t>(mtp_fc->dimensions[0]),
          static_cast<std::size_t>(mtp_fc->dimensions[1]), 1,
          sequential_w3, stream);
    } else {
      status = matrix(mtp_fc, convolved_qkv, projected_main);
      if (!status.ok()) return status;
      status = launch_float_to_half(projected_main, gdn_output, kHidden, stream);
    }
    if (!status.ok()) return status;

    const auto& layer = mtp_layer;
    const __half* mtp_input = gdn_output;
    status = launch_rms_norm(
        mtp_input, reinterpret_cast<const std::uint16_t*>(layer.input_norm->data),
        normalized, kHidden, Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!status.ok()) return status;
    const std::array<const DeviceTensorView*, 3> qkv_weights{
        layer.q, layer.k, layer.v};
    const std::array<float*, 3> qkv_outputs{
        projected_main, projected_k, projected_v};
    status = matrices(qkv_weights, qkv_outputs, normalized);
    if (!status.ok()) return status;
    status = device_position == nullptr
                 ? launch_qk_rms_norm_rope_block(
                       projected_main,
                       reinterpret_cast<const std::uint16_t*>(
                           layer.q_norm->data),
                       query, 24, 512, projected_k,
                       reinterpret_cast<const std::uint16_t*>(
                           layer.k_norm->data),
                       key, 4, 256, 256, Qwen38Spec::rotary_size, position, 1,
                       Qwen38Spec::rope_theta, stream)
                 : launch_qk_rms_norm_rope_block_indirect_position(
                       projected_main,
                       reinterpret_cast<const std::uint16_t*>(
                           layer.q_norm->data),
                       query, 24, 512, projected_k,
                       reinterpret_cast<const std::uint16_t*>(
                           layer.k_norm->data),
                       key, 4, 256, 256, Qwen38Spec::rotary_size,
                       device_position, 1, Qwen38Spec::rope_theta, stream);
    if (!status.ok()) return status;
    status = device_position == nullptr
                 ? launch_kv_append(
                       key, projected_v, mtp_key_cache, mtp_value_cache,
                       position, config.maximum_context_tokens, stream)
                 : launch_kv_append_block_indirect_position(
                       key, projected_v, mtp_key_cache, mtp_value_cache,
                       device_position, config.maximum_context_tokens, 1,
                       stream);
    if (!status.ok()) return status;
    // A fully accepted speculative block only needs to advance the MTP
    // attention cache through its final draft. No downstream attention, MLP,
    // proposal-head, or argmax work contributes to the next round.
    if (output_token == nullptr) return Status::ok_status();
    status = device_position == nullptr
                 ? launch_gated_attention_decode(
                       query, projected_main, mtp_key_cache, mtp_value_cache,
                       attention_output, position + 1,
                       config.maximum_context_tokens, stream)
                 : launch_gated_attention_decode_block_indirect_position(
                       query, projected_main, mtp_key_cache, mtp_value_cache,
                       attention_output, device_position,
                       config.maximum_context_tokens, 1, stream);
    if (!status.ok()) return status;
    status = matrix(layer.out, attention_output, mixer_update);
    if (!status.ok()) return status;
    status = launch_residual_add_rms_norm_batch(
        mtp_input, mixer_update, post_attention,
        reinterpret_cast<const std::uint16_t*>(layer.post_norm->data),
        post_normalized, kHidden, 1, Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!status.ok()) return status;
    const std::array<const DeviceTensorView*, 2> mlp_weights{
        layer.mlp_gate, layer.mlp_up};
    const std::array<float*, 2> mlp_outputs{gate_output, up_output};
    status = matrices(mlp_weights, mlp_outputs, post_normalized);
    if (!status.ok()) return status;
    status = launch_mlp_down(layer.mlp_down);
    if (!status.ok()) return status;
    status = launch_residual_add_rms_norm_batch(
        post_attention, mlp_update, mtp_hidden,
        reinterpret_cast<const std::uint16_t*>(mtp_final_norm->data),
        normalized, kHidden, 1, Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!status.ok()) return status;
    status = matrix(&mtp_draft_head, normalized, logits);
    if (!status.ok()) return status;
    return mtp_draft_token_ids != nullptr
               ? launch_argmax_mapped(
                     logits,
                     reinterpret_cast<const std::uint32_t*>(mtp_draft_token_ids->data),
                     output_token, kMtpDraftVocabulary, stream)
               : launch_argmax(logits, output_token, kMtpDraftVocabulary, stream);
  }

  [[nodiscard]] Status launch_speculative_verifier(
      std::uint32_t seed_token,
      const std::uint32_t* device_drafts,
      unsigned draft_count) {
    if (device_drafts == nullptr || draft_count == 0 ||
        draft_count + 1 > kMaximumBlock) {
      return {ErrorCode::invalid_argument, "invalid speculative verifier inputs"};
    }
    const unsigned batch = draft_count + 1;
    auto status = enqueue_verifier_body(
        seed_token, nullptr, device_drafts, draft_count, nullptr);
    if (!status.ok()) return status;
    // Qwen3.8 has an even 64-layer stack, so the verifier's final residual
    // returns to hidden_a for every batch.
    last_hidden = hidden_a + static_cast<std::size_t>(batch - 1) * kHidden;
    current_position += batch;
    return Status::ok_status();
  }

  [[nodiscard]] Status launch_prepared_speculative_graph(
      std::uint32_t seed_token,
      std::size_t base_target_position,
      std::size_t base_mtp_position) {
    const unsigned batch = config.graph_verifier_batch;
    if (batch < 2U) {
      return {ErrorCode::internal, "prepared speculative graph is disabled"};
    }
    const unsigned draft_count = batch - 1U;
    graph_host_seed_token = seed_token;
    graph_host_base_position = base_target_position;
    for (unsigned index = 0; index < draft_count; ++index) {
      graph_host_mtp_positions[index] = base_mtp_position + index;
    }

    const auto stream = engine->compute_stream();
    if (const auto error = hipMemcpyAsync(
            graph_seed_token, &graph_host_seed_token,
            sizeof(graph_host_seed_token), hipMemcpyHostToDevice, stream);
        error != hipSuccess) {
      return hip_status(error, "speculative graph seed upload failed");
    }
    if (const auto error = hipMemcpyAsync(
            graph_base_position, &graph_host_base_position,
            sizeof(graph_host_base_position), hipMemcpyHostToDevice, stream);
        error != hipSuccess) {
      return hip_status(error, "speculative graph verifier-position upload failed");
    }
    if (const auto error = hipMemcpyAsync(
            graph_mtp_positions, graph_host_mtp_positions.data(),
            draft_count * sizeof(std::uint64_t), hipMemcpyHostToDevice, stream);
        error != hipSuccess) {
      return hip_status(error, "speculative graph MTP-position upload failed");
    }

    const auto& graph = speculative_graphs[gdn_state_parity ? 1U : 0U];
    if (graph.executable == nullptr) {
      return {ErrorCode::internal, "prepared speculative graph is unavailable"};
    }
    if (const auto error = hipGraphLaunch(graph.executable, stream);
        error != hipSuccess) {
      return hip_status(error, "speculative graph launch failed");
    }
    last_hidden = hidden_a + static_cast<std::size_t>(draft_count) * kHidden;
    current_position += batch;
    return Status::ok_status();
  }
};

QwenDecodeSession::QwenDecodeSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
QwenDecodeSession::~QwenDecodeSession() = default;
QwenDecodeSession::QwenDecodeSession(QwenDecodeSession&&) noexcept = default;
QwenDecodeSession& QwenDecodeSession::operator=(QwenDecodeSession&&) noexcept = default;

Result<QwenDecodeSession> QwenDecodeSession::create(
    Engine& engine,
    DecodeSessionConfig config) {
  if (config.maximum_context_tokens == 0 ||
      config.maximum_context_tokens > Qwen38Spec::maximum_context_tokens ||
      (config.graph_verifier_batch != 0 &&
       (config.graph_verifier_batch < 2 ||
        config.graph_verifier_batch > kMaximumBlock)) ||
      engine.compute_stream() == nullptr) {
    return Status{ErrorCode::invalid_argument, "invalid decode session configuration"};
  }
  auto impl = std::make_unique<Impl>();
  impl->engine = &engine;
  impl->config = config;
  impl->tiled_weights = std::string_view(
      engine.load_summary().identity.recipe_id).find("tiled-v") != std::string_view::npos;
  impl->sequential_w3 = std::string_view(
      engine.load_summary().identity.recipe_id).find("w3seq") != std::string_view::npos;
  impl->embedding = engine.find_device_tensor("model.language_model.embed_tokens.weight");
  impl->final_norm = engine.find_device_tensor("model.language_model.norm.weight");
  impl->lm_head = engine.find_device_tensor("lm_head.weight");
  if (impl->embedding == nullptr || impl->final_norm == nullptr || impl->lm_head == nullptr) {
    return Status{ErrorCode::corrupt_artifact, "artifact lacks embedding, final norm, or LM head"};
  }
  if (impl->lm_head->rank != 2 ||
      impl->lm_head->dimensions[0] < kMtpDraftVocabulary) {
    return Status{ErrorCode::corrupt_artifact,
                  "artifact LM head is too small for the MTP shortlist"};
  }
  const auto* ranked_draft_head = engine.find_device_tensor("mtp.draft_head.weight");
  impl->mtp_draft_token_ids = engine.find_device_tensor("mtp.draft_head_token_ids");
  if ((ranked_draft_head == nullptr) != (impl->mtp_draft_token_ids == nullptr)) {
    return Status{ErrorCode::corrupt_artifact,
                  "ranked MTP head and token-ID map must be present together"};
  }
  if (ranked_draft_head != nullptr) {
    if (ranked_draft_head->rank != 2 ||
        ranked_draft_head->dimensions[0] < kMtpDraftVocabulary ||
        ranked_draft_head->dimensions[1] != kHidden ||
        impl->mtp_draft_token_ids->rank != 1 ||
        impl->mtp_draft_token_ids->dimensions[0] !=
            ranked_draft_head->dimensions[0] ||
        impl->mtp_draft_token_ids->dtype != DType::i32 ||
        impl->mtp_draft_token_ids->quantization != Quantization::none) {
      return Status{ErrorCode::corrupt_artifact,
                    "ranked MTP head or token-ID map has an invalid schema"};
    }
    impl->mtp_draft_head = *ranked_draft_head;
    impl->mtp_draft_head.dimensions[0] = kMtpDraftVocabulary;
    impl->mtp_draft_head.data_bytes =
        ranked_draft_head->data_bytes * kMtpDraftVocabulary /
        ranked_draft_head->dimensions[0];
    impl->mtp_draft_head.auxiliary_bytes =
        ranked_draft_head->auxiliary_bytes * kMtpDraftVocabulary /
        ranked_draft_head->dimensions[0];
  } else {
    impl->mtp_draft_head = *impl->lm_head;
    impl->mtp_draft_head.name = "lm_head.weight[mtp-prefix]";
    impl->mtp_draft_head.dimensions[0] = kMtpDraftVocabulary;
    impl->mtp_draft_head.data_bytes =
        impl->lm_head->data_bytes * kMtpDraftVocabulary /
        impl->lm_head->dimensions[0];
    impl->mtp_draft_head.auxiliary_bytes =
        impl->lm_head->auxiliary_bytes * kMtpDraftVocabulary /
        impl->lm_head->dimensions[0];
  }

  impl->layers.resize(Qwen38Spec::layer_count);
  std::size_t gdn_index = 0;
  std::size_t attention_index = 0;
  for (std::uint32_t index = 0; index < Qwen38Spec::layer_count; ++index) {
    auto& layer = impl->layers[index];
    const auto prefix = "model.language_model.layers." + std::to_string(index) + ".";
    auto find = [&](std::string_view suffix) {
      return engine.find_device_tensor(prefix + std::string(suffix));
    };
    layer.attention = Qwen38Spec::is_full_attention(index);
    layer.state_index = layer.attention ? attention_index++ : gdn_index++;
    layer.input_norm = find("input_layernorm.weight");
    layer.post_norm = find("post_attention_layernorm.weight");
    layer.mlp_gate = find("mlp.gate_proj.weight");
    layer.mlp_up = find("mlp.up_proj.weight");
    layer.mlp_down = find("mlp.down_proj.weight");
    if (layer.attention) {
      layer.q = find("self_attn.q_proj.weight");
      layer.k = find("self_attn.k_proj.weight");
      layer.v = find("self_attn.v_proj.weight");
      layer.q_norm = find("self_attn.q_norm.weight");
      layer.k_norm = find("self_attn.k_norm.weight");
      layer.out = find("self_attn.o_proj.weight");
    } else {
      layer.qkv = find("linear_attn.in_proj_qkv.weight");
      layer.z = find("linear_attn.in_proj_z.weight");
      layer.a = find("linear_attn.in_proj_a.weight");
      layer.b = find("linear_attn.in_proj_b.weight");
      layer.convolution = find("linear_attn.conv1d.weight");
      layer.a_log = find("linear_attn.A_log");
      layer.dt_bias = find("linear_attn.dt_bias");
      layer.gdn_norm = find("linear_attn.norm.weight");
      layer.out = find("linear_attn.out_proj.weight");
    }
    const DeviceTensorView* common[] = {
        layer.input_norm, layer.post_norm, layer.mlp_gate, layer.mlp_up,
        layer.mlp_down, layer.out,
    };
    if (std::any_of(std::begin(common), std::end(common),
                    [](const auto* value) { return value == nullptr; }) ||
        (layer.attention &&
         (layer.q == nullptr || layer.k == nullptr || layer.v == nullptr ||
          layer.q_norm == nullptr || layer.k_norm == nullptr)) ||
        (!layer.attention &&
         (layer.qkv == nullptr || layer.z == nullptr || layer.a == nullptr ||
          layer.b == nullptr || layer.convolution == nullptr || layer.a_log == nullptr ||
          layer.dt_bias == nullptr || layer.gdn_norm == nullptr))) {
      return Status{ErrorCode::corrupt_artifact,
                    "artifact lacks a required tensor for decoder layer " +
                        std::to_string(index)};
    }
  }
  if (gdn_index != kGdnLayerCount || attention_index != kAttentionLayerCount) {
    return Status{ErrorCode::internal, "decoder layer topology is inconsistent"};
  }

  auto mtp_find = [&](std::string_view suffix) {
    return engine.find_device_tensor("mtp." + std::string(suffix));
  };
  impl->mtp_fc = mtp_find("fc.weight");
  impl->mtp_pre_embedding_norm = mtp_find("pre_fc_norm_embedding.weight");
  impl->mtp_pre_hidden_norm = mtp_find("pre_fc_norm_hidden.weight");
  impl->mtp_final_norm = mtp_find("norm.weight");
  auto& mtp = impl->mtp_layer;
  mtp.attention = true;
  mtp.input_norm = mtp_find("layers.0.input_layernorm.weight");
  mtp.post_norm = mtp_find("layers.0.post_attention_layernorm.weight");
  mtp.mlp_gate = mtp_find("layers.0.mlp.gate_proj.weight");
  mtp.mlp_up = mtp_find("layers.0.mlp.up_proj.weight");
  mtp.mlp_down = mtp_find("layers.0.mlp.down_proj.weight");
  mtp.q = mtp_find("layers.0.self_attn.q_proj.weight");
  mtp.k = mtp_find("layers.0.self_attn.k_proj.weight");
  mtp.v = mtp_find("layers.0.self_attn.v_proj.weight");
  mtp.q_norm = mtp_find("layers.0.self_attn.q_norm.weight");
  mtp.k_norm = mtp_find("layers.0.self_attn.k_norm.weight");
  mtp.out = mtp_find("layers.0.self_attn.o_proj.weight");
  const DeviceTensorView* mtp_required[] = {
      impl->mtp_fc, impl->mtp_pre_embedding_norm, impl->mtp_pre_hidden_norm,
      impl->mtp_final_norm, mtp.input_norm, mtp.post_norm, mtp.mlp_gate,
      mtp.mlp_up, mtp.mlp_down, mtp.q, mtp.k, mtp.v, mtp.q_norm,
      mtp.k_norm, mtp.out,
  };
  if (std::any_of(std::begin(mtp_required), std::end(mtp_required),
                  [](const auto* value) { return value == nullptr; })) {
    return Status{ErrorCode::corrupt_artifact, "artifact lacks a required MTP tensor"};
  }

  if (config.maximum_context_tokens >
      std::numeric_limits<std::size_t>::max() /
          (kAttentionLayerCount * kKvElements * sizeof(__half))) {
    return Status{ErrorCode::out_of_memory, "KV cache size overflow"};
  }
  const auto per_cache_bytes = kAttentionLayerCount * config.maximum_context_tokens *
                               kKvElements * sizeof(__half);
  const auto mtp_per_cache_bytes =
      config.maximum_context_tokens * kKvElements * sizeof(__half);
  const auto conv_bytes = kGdnLayerCount * kConvStateElements * sizeof(__half);
  const auto gdn_bytes = kGdnLayerCount * kGdnStateElements * sizeof(float);
  const auto conv_log_bytes =
      kGdnLayerCount * kMaximumBlock * kGdnQkv * sizeof(__half);
  const auto gdn_key_log_bytes =
      kGdnLayerCount * kMaximumBlock * kGdnKey * sizeof(float);
  const auto gdn_delta_log_bytes =
      kGdnLayerCount * kMaximumBlock * kGdnValue * sizeof(float);
  const auto gdn_decay_log_bytes =
      kGdnLayerCount * kMaximumBlock * kGdnHeads * sizeof(float);

  std::size_t cursor = 0;
  auto reserve = [&](std::size_t bytes, std::size_t& offset) {
    constexpr std::size_t alignment = 256;
    const auto mask = alignment - 1;
    if (cursor > std::numeric_limits<std::size_t>::max() - mask) return false;
    cursor = (cursor + mask) & ~mask;
    if (bytes > std::numeric_limits<std::size_t>::max() - cursor) return false;
    offset = cursor;
    cursor += bytes;
    return true;
  };
  std::size_t hidden_a_offset, hidden_b_offset, mtp_hidden_offset;
  std::size_t target_hidden_snapshot_offset;
  std::size_t normalized_offset, convolved_offset;
  std::size_t gdn_output_offset, query_offset, key_offset, attention_output_offset;
  std::size_t post_attention_offset, post_normalized_offset, activated_offset;
  std::size_t projected_main_offset, projected_z_offset, projected_a_offset;
  std::size_t projected_b_offset, projected_k_offset, projected_v_offset;
  std::size_t mixer_update_offset, gate_output_offset, up_output_offset;
  std::size_t mlp_update_offset, logits_offset, selected_token_offset;
  std::size_t graph_base_position_offset, graph_mtp_positions_offset;
  std::size_t conv_states_offset, gdn_states_offset;
  std::size_t conv_snapshot_offset, gdn_snapshot_offset;
  std::size_t conv_rollback_log_offset, gdn_key_log_offset;
  std::size_t gdn_delta_log_offset, gdn_decay_log_offset;
  std::size_t key_caches_offset, value_caches_offset;
  std::size_t mtp_key_cache_offset, mtp_value_cache_offset;
  std::size_t activation_codes_offset, activation_scales_offset;
  const bool layout_ok =
      reserve(kMaximumBlock * kHidden * sizeof(__half), hidden_a_offset) &&
      reserve(kMaximumBlock * kHidden * sizeof(__half), hidden_b_offset) &&
      reserve(kHidden * sizeof(__half), mtp_hidden_offset) &&
      reserve(kHidden * sizeof(__half), target_hidden_snapshot_offset) &&
      reserve(kMaximumBlock * kHidden * sizeof(__half), normalized_offset) &&
      reserve(kMaximumBlock * kGdnQkv * sizeof(__half), convolved_offset) &&
      reserve(kMaximumBlock * kGdnValue * sizeof(__half), gdn_output_offset) &&
      reserve(kMaximumBlock * kQueryElements * sizeof(__half), query_offset) &&
      reserve(kMaximumBlock * kKvElements * sizeof(__half), key_offset) &&
      reserve(kMaximumBlock * kQueryElements * sizeof(__half), attention_output_offset) &&
      reserve(kMaximumBlock * kHidden * sizeof(__half), post_attention_offset) &&
      reserve(kMaximumBlock * kHidden * sizeof(__half), post_normalized_offset) &&
      reserve(kMaximumBlock * kIntermediate * sizeof(__half), activated_offset) &&
      reserve(kMaximumBlock * kQueryProjectionElements * sizeof(float), projected_main_offset) &&
      reserve(kMaximumBlock * kGdnValue * sizeof(float), projected_z_offset) &&
      reserve(kMaximumBlock * kGdnHeads * sizeof(float), projected_a_offset) &&
      reserve(kMaximumBlock * kGdnHeads * sizeof(float), projected_b_offset) &&
      reserve(kMaximumBlock * kKvElements * sizeof(float), projected_k_offset) &&
      reserve(kMaximumBlock * kKvElements * sizeof(float), projected_v_offset) &&
      reserve(kMaximumBlock * kHidden * sizeof(float), mixer_update_offset) &&
      reserve(kMaximumBlock * kIntermediate * sizeof(float), gate_output_offset) &&
      reserve(kMaximumBlock * kIntermediate * sizeof(float), up_output_offset) &&
      reserve(kMaximumBlock * kHidden * sizeof(float), mlp_update_offset) &&
      reserve(kMaximumBlock * Qwen38Spec::vocabulary_size * sizeof(float), logits_offset) &&
      reserve(2U * kMaximumBlock * sizeof(std::uint32_t), selected_token_offset) &&
      reserve(sizeof(std::uint64_t), graph_base_position_offset) &&
      reserve(kMaximumBlock * sizeof(std::uint64_t), graph_mtp_positions_offset) &&
      reserve(kMaximumBlock * kIntermediate, activation_codes_offset) &&
      reserve(kMaximumBlock * (kIntermediate / 128) * sizeof(__half), activation_scales_offset) &&
      reserve(conv_bytes, conv_states_offset) && reserve(gdn_bytes, gdn_states_offset) &&
      reserve(conv_bytes, conv_snapshot_offset) && reserve(gdn_bytes, gdn_snapshot_offset) &&
      reserve(conv_log_bytes, conv_rollback_log_offset) &&
      reserve(gdn_key_log_bytes, gdn_key_log_offset) &&
      reserve(gdn_delta_log_bytes, gdn_delta_log_offset) &&
      reserve(gdn_decay_log_bytes, gdn_decay_log_offset) &&
      reserve(per_cache_bytes, key_caches_offset) && reserve(per_cache_bytes, value_caches_offset) &&
      reserve(mtp_per_cache_bytes, mtp_key_cache_offset) &&
      reserve(mtp_per_cache_bytes, mtp_value_cache_offset);
  if (!layout_ok) return Status{ErrorCode::out_of_memory, "decode arena layout overflow"};

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (const auto error = hipMemGetInfo(&free_bytes, &total_bytes); error != hipSuccess)
    return hip_status(error, "hipMemGetInfo before session allocation failed");
  if (cursor > free_bytes) {
    return Status{ErrorCode::out_of_memory,
                  "decode session arena exceeds currently free GPU memory"};
  }
  if (const auto error = hipMalloc(reinterpret_cast<void**>(&impl->arena), cursor);
      error != hipSuccess) {
    return hip_status(error, "decode session arena allocation failed");
  }
  auto pointer = [&](std::size_t offset) { return impl->arena + offset; };
  impl->hidden_a = reinterpret_cast<__half*>(pointer(hidden_a_offset));
  impl->hidden_b = reinterpret_cast<__half*>(pointer(hidden_b_offset));
  impl->mtp_hidden = reinterpret_cast<__half*>(pointer(mtp_hidden_offset));
  impl->target_hidden_snapshot =
      reinterpret_cast<__half*>(pointer(target_hidden_snapshot_offset));
  impl->normalized = reinterpret_cast<__half*>(pointer(normalized_offset));
  impl->convolved_qkv = reinterpret_cast<__half*>(pointer(convolved_offset));
  impl->gdn_output = reinterpret_cast<__half*>(pointer(gdn_output_offset));
  impl->query = reinterpret_cast<__half*>(pointer(query_offset));
  impl->key = reinterpret_cast<__half*>(pointer(key_offset));
  impl->attention_output = reinterpret_cast<__half*>(pointer(attention_output_offset));
  impl->post_attention = reinterpret_cast<__half*>(pointer(post_attention_offset));
  impl->post_normalized = reinterpret_cast<__half*>(pointer(post_normalized_offset));
  impl->activated = reinterpret_cast<__half*>(pointer(activated_offset));
  impl->projected_main = reinterpret_cast<float*>(pointer(projected_main_offset));
  impl->projected_z = reinterpret_cast<float*>(pointer(projected_z_offset));
  impl->projected_a = reinterpret_cast<float*>(pointer(projected_a_offset));
  impl->projected_b = reinterpret_cast<float*>(pointer(projected_b_offset));
  impl->projected_k = reinterpret_cast<float*>(pointer(projected_k_offset));
  impl->projected_v = reinterpret_cast<float*>(pointer(projected_v_offset));
  impl->mixer_update = reinterpret_cast<float*>(pointer(mixer_update_offset));
  impl->gate_output = reinterpret_cast<float*>(pointer(gate_output_offset));
  impl->up_output = reinterpret_cast<float*>(pointer(up_output_offset));
  impl->mlp_update = reinterpret_cast<float*>(pointer(mlp_update_offset));
  impl->logits = reinterpret_cast<float*>(pointer(logits_offset));
  impl->selected_token = reinterpret_cast<std::uint32_t*>(pointer(selected_token_offset));
  impl->draft_tokens = impl->selected_token + kMaximumBlock;
  impl->graph_seed_token = impl->draft_tokens + kMaximumBlock - 1U;
  impl->graph_base_position =
      reinterpret_cast<std::uint64_t*>(pointer(graph_base_position_offset));
  impl->graph_mtp_positions =
      reinterpret_cast<std::uint64_t*>(pointer(graph_mtp_positions_offset));
  impl->activation_codes = reinterpret_cast<std::uint8_t*>(pointer(activation_codes_offset));
  impl->activation_scales = reinterpret_cast<__half*>(pointer(activation_scales_offset));
  impl->conv_states = reinterpret_cast<__half*>(pointer(conv_states_offset));
  impl->gdn_states = reinterpret_cast<float*>(pointer(gdn_states_offset));
  impl->conv_snapshot = reinterpret_cast<__half*>(pointer(conv_snapshot_offset));
  impl->gdn_snapshot = reinterpret_cast<float*>(pointer(gdn_snapshot_offset));
  impl->conv_rollback_log =
      reinterpret_cast<__half*>(pointer(conv_rollback_log_offset));
  impl->gdn_key_log = reinterpret_cast<float*>(pointer(gdn_key_log_offset));
  impl->gdn_delta_log = reinterpret_cast<float*>(pointer(gdn_delta_log_offset));
  impl->gdn_decay_log = reinterpret_cast<float*>(pointer(gdn_decay_log_offset));
  impl->key_caches = reinterpret_cast<__half*>(pointer(key_caches_offset));
  impl->value_caches = reinterpret_cast<__half*>(pointer(value_caches_offset));
  impl->mtp_key_cache = reinterpret_cast<__half*>(pointer(mtp_key_cache_offset));
  impl->mtp_value_cache = reinterpret_cast<__half*>(pointer(mtp_value_cache_offset));
  const auto stream = engine.compute_stream();
  if (auto error = hipMemsetAsync(impl->conv_states, 0, conv_bytes, stream);
      error != hipSuccess) {
    return hip_status(error, "convolution state initialization failed");
  }
  if (auto error = hipMemsetAsync(impl->gdn_states, 0, gdn_bytes, stream);
      error != hipSuccess) {
    return hip_status(error, "GDN state initialization failed");
  }
  if (auto error = hipStreamSynchronize(stream); error != hipSuccess)
    return hip_status(error, "decode session initialization failed");
  auto graph_status = impl->prepare_speculative_graphs();
  if (!graph_status.ok()) return graph_status;
  impl->summary = {cursor, config.maximum_context_tokens, conv_bytes + gdn_bytes,
                   per_cache_bytes * 2, mtp_per_cache_bytes * 2};
  return QwenDecodeSession(std::move(impl));
}

Result<std::uint32_t> QwenDecodeSession::greedy_step(std::uint32_t input_token) {
  if (!impl_ || impl_->current_position >= impl_->config.maximum_context_tokens ||
      input_token >= Qwen38Spec::vocabulary_size) {
    return Status{ErrorCode::invalid_argument, "decode token or session position is invalid"};
  }
  const auto stream = impl_->engine->compute_stream();
  auto status = impl_->tiled_weights
                    ? launch_lowbit_row_gather_tiled(
                          impl_->embedding->quantization,
                          reinterpret_cast<const std::uint8_t*>(impl_->embedding->data),
                          reinterpret_cast<const __half*>(impl_->embedding->auxiliary),
                          impl_->hidden_a, input_token,
                          static_cast<std::size_t>(impl_->embedding->dimensions[0]), kHidden,
                          stream)
                    : launch_lowbit_row_gather(
                          impl_->embedding->quantization,
                          reinterpret_cast<const std::uint8_t*>(impl_->embedding->data),
                          reinterpret_cast<const __half*>(impl_->embedding->auxiliary),
                          impl_->hidden_a, input_token,
                          static_cast<std::size_t>(impl_->embedding->dimensions[0]), kHidden,
                          stream);
  if (!status.ok()) return status;
  __half* current = impl_->hidden_a;
  __half* next = impl_->hidden_b;
  status = launch_rms_norm(
      current,
      reinterpret_cast<const std::uint16_t*>(impl_->layers.front().input_norm->data),
      impl_->normalized, kHidden, Qwen38Spec::rms_norm_epsilon, true, stream);
  if (!status.ok()) return status;
  for (std::size_t index = 0; index < impl_->layers.size(); ++index) {
    const auto* next_norm = reinterpret_cast<const std::uint16_t*>(
        index + 1 < impl_->layers.size()
            ? impl_->layers[index + 1].input_norm->data
            : impl_->final_norm->data);
    status = impl_->launch_layer(impl_->layers[index], current, next, next_norm);
    if (!status.ok()) return status;
    std::swap(current, next);
  }
  impl_->last_hidden = current;
  status = impl_->matrix(impl_->lm_head, impl_->normalized, impl_->logits);
  if (!status.ok()) return status;
  status = launch_argmax(
      impl_->logits, impl_->selected_token, Qwen38Spec::tokenizer_vocabulary_size, stream);
  if (!status.ok()) return status;
  std::uint32_t output_token = 0;
  if (const auto error = hipMemcpyAsync(&output_token, impl_->selected_token,
                                        sizeof(output_token), hipMemcpyDeviceToHost, stream);
      error != hipSuccess) {
    return hip_status(error, "selected token download failed");
  }
  if (const auto error = hipStreamSynchronize(stream); error != hipSuccess)
    return hip_status(error, "decode step failed");
  ++impl_->current_position;
  return output_token;
}

Result<std::vector<std::uint32_t>> QwenDecodeSession::block_step(
    std::span<const std::uint32_t> input_tokens) {
  if (!impl_ || input_tokens.empty() || input_tokens.size() > kMaximumBlock ||
      (!impl_->tiled_weights && input_tokens.size() > 8) ||
      input_tokens.size() > impl_->config.maximum_context_tokens - impl_->current_position ||
      std::any_of(input_tokens.begin(), input_tokens.end(), [](std::uint32_t token) {
        return token >= Qwen38Spec::vocabulary_size;
      })) {
    return Status{ErrorCode::invalid_argument,
                  "decode block must contain a supported number of valid tokens within the context limit"};
  }
  const auto stream = impl_->engine->compute_stream();
  const auto batch = static_cast<unsigned>(input_tokens.size());
  Status status;
  for (unsigned token = 0; token < batch; ++token) {
    auto* destination = impl_->hidden_a + static_cast<std::size_t>(token) * kHidden;
    status = impl_->tiled_weights
                 ? launch_lowbit_row_gather_tiled(
                       impl_->embedding->quantization,
                       reinterpret_cast<const std::uint8_t*>(impl_->embedding->data),
                       reinterpret_cast<const __half*>(impl_->embedding->auxiliary),
                       destination, input_tokens[token],
                       static_cast<std::size_t>(impl_->embedding->dimensions[0]),
                       kHidden, stream)
                 : launch_lowbit_row_gather(
                       impl_->embedding->quantization,
                       reinterpret_cast<const std::uint8_t*>(impl_->embedding->data),
                       reinterpret_cast<const __half*>(impl_->embedding->auxiliary),
                       destination, input_tokens[token],
                       static_cast<std::size_t>(impl_->embedding->dimensions[0]),
                       kHidden, stream);
    if (!status.ok()) return status;
  }

  __half* current = impl_->hidden_a;
  __half* next = impl_->hidden_b;
  status = launch_rms_norm_batch(
      current,
      reinterpret_cast<const std::uint16_t*>(impl_->layers.front().input_norm->data),
      impl_->normalized, kHidden, batch,
      Qwen38Spec::rms_norm_epsilon, true, stream);
  if (!status.ok()) return status;
  for (std::size_t index = 0; index < impl_->layers.size(); ++index) {
    const auto* next_norm = reinterpret_cast<const std::uint16_t*>(
        index + 1 < impl_->layers.size()
            ? impl_->layers[index + 1].input_norm->data
            : impl_->final_norm->data);
    status = impl_->launch_layer_block(
        impl_->layers[index], current, next, batch, next_norm);
    if (!status.ok()) return status;
    std::swap(current, next);
  }
  impl_->last_hidden = current + (input_tokens.size() - 1U) * kHidden;
  status = impl_->matrix(impl_->lm_head, impl_->normalized, impl_->logits, batch);
  if (!status.ok()) return status;
  status = launch_argmax_batch(
      impl_->logits, impl_->selected_token,
      Qwen38Spec::tokenizer_vocabulary_size,
      Qwen38Spec::vocabulary_size, batch, stream);
  if (!status.ok()) return status;
  std::vector<std::uint32_t> output_tokens(batch);
  if (const auto error = hipMemcpyAsync(
          output_tokens.data(), impl_->selected_token,
          output_tokens.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost, stream);
      error != hipSuccess) {
    return hip_status(error, "decode block token download failed");
  }
  if (const auto error = hipStreamSynchronize(stream); error != hipSuccess) {
    return hip_status(error, "decode block failed");
  }
  impl_->current_position += input_tokens.size();
  return output_tokens;
}

Result<std::uint32_t> QwenDecodeSession::mtp_draft(std::uint32_t next_token) {
  if (!impl_ || impl_->last_hidden == nullptr ||
      (!impl_->mtp_chain_active &&
       impl_->mtp_position + 1 != impl_->current_position) ||
      next_token >= Qwen38Spec::tokenizer_vocabulary_size) {
    return Status{ErrorCode::invalid_argument,
                  "MTP drafting is not aligned with the target state"};
  }
  const __half* conditioning_hidden =
      impl_->mtp_chain_active ? impl_->mtp_hidden : impl_->last_hidden;
  auto status = impl_->launch_mtp_proposal(
      next_token, nullptr, conditioning_hidden, impl_->mtp_position,
      impl_->selected_token);
  if (!status.ok()) return status;
  const auto stream = impl_->engine->compute_stream();
  std::uint32_t output_token = 0;
  if (const auto error = hipMemcpyAsync(&output_token, impl_->selected_token,
                                        sizeof(output_token), hipMemcpyDeviceToHost, stream);
      error != hipSuccess) {
    return hip_status(error, "MTP selected token download failed");
  }
  if (const auto error = hipStreamSynchronize(stream); error != hipSuccess)
    return hip_status(error, "MTP draft step failed");
  ++impl_->mtp_position;
  return output_token;
}

Result<SpeculativeStepResult> QwenDecodeSession::speculative_step(
    std::uint32_t seed_token,
    std::size_t draft_count) {
  if (!impl_ || impl_->last_hidden == nullptr || draft_count == 0 ||
      draft_count >= kMaximumBlock ||
      seed_token >= Qwen38Spec::tokenizer_vocabulary_size ||
      impl_->mtp_position + 1 != impl_->current_position ||
      draft_count + 1 >
          impl_->config.maximum_context_tokens - impl_->current_position) {
    return Status{ErrorCode::invalid_argument,
                  "speculative step requires 1..31 drafts and aligned target/MTP state"};
  }

  const auto base_target_position = impl_->current_position;
  const auto base_mtp_position = impl_->mtp_position;
  auto status = impl_->snapshot_target_state();
  if (!status.ok()) return status;

  const auto stream = impl_->engine->compute_stream();
  std::vector<std::uint32_t> drafts(draft_count);
  const bool use_prepared_graph =
      draft_count + 1U == impl_->config.graph_verifier_batch;
  if (use_prepared_graph) {
    status = impl_->launch_prepared_speculative_graph(
        seed_token, base_target_position, base_mtp_position);
  } else {
    for (std::size_t index = 0; index < draft_count; ++index) {
      const auto* indirect_token =
          index == 0 ? nullptr : impl_->draft_tokens + index - 1;
      const auto* conditioning_hidden =
          index == 0 ? impl_->target_hidden_snapshot : impl_->mtp_hidden;
      status = impl_->launch_mtp_proposal(
          seed_token, indirect_token, conditioning_hidden,
          base_mtp_position + index, impl_->draft_tokens + index);
      if (!status.ok()) break;
    }
    if (status.ok()) {
      impl_->record_rollback = true;
      status = impl_->launch_speculative_verifier(
          seed_token, impl_->draft_tokens, static_cast<unsigned>(draft_count));
      impl_->record_rollback = false;
    }
  }
  if (!status.ok()) {
    impl_->record_rollback = false;
    impl_->mtp_chain_active = false;
    impl_->mtp_position = base_mtp_position;
    impl_->current_position = base_target_position;
    const auto restore = impl_->restore_target_state();
    if (!restore.ok()) return restore;
    if (const auto error = hipStreamSynchronize(impl_->engine->compute_stream());
        error != hipSuccess) {
      return hip_status(error, "speculative failure rollback failed");
    }
    return status;
  }
  impl_->mtp_position = base_mtp_position + draft_count;
  impl_->mtp_chain_active = true;

  std::vector<std::uint32_t> target_outputs(draft_count + 1);
  if (const auto error = hipMemcpyAsync(
          drafts.data(), impl_->draft_tokens,
          drafts.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost, stream);
      error != hipSuccess) {
    return hip_status(error, "MTP proposal-chain download failed");
  }
  if (const auto error = hipMemcpyAsync(
          target_outputs.data(), impl_->selected_token,
          target_outputs.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost, stream);
      error != hipSuccess) {
    return hip_status(error, "speculative verifier download failed");
  }
  if (const auto error = hipStreamSynchronize(stream); error != hipSuccess) {
    return hip_status(error, "speculative proposal/verifier chain failed");
  }

  std::size_t accepted = 0;
  while (accepted < draft_count &&
         drafts[accepted] == target_outputs[accepted]) {
    ++accepted;
  }

  if (accepted != draft_count) {
    status = impl_->restore_target_state();
    if (!status.ok()) {
      impl_->mtp_chain_active = false;
      return status;
    }
    status = impl_->restore_logged_prefix(static_cast<unsigned>(accepted + 1));
    if (!status.ok()) {
      impl_->mtp_chain_active = false;
      return status;
    }
    impl_->current_position = base_target_position + accepted + 1;
    impl_->mtp_position = base_mtp_position + accepted + 1;
    impl_->mtp_chain_active = false;
    impl_->last_hidden = impl_->hidden_a + accepted * kHidden;
    // Restoration is ordered before the next decode operation on the same
    // stream. No host-visible value depends on waiting for it here.
  } else {
    std::swap(impl_->conv_states, impl_->conv_snapshot);
    std::swap(impl_->gdn_states, impl_->gdn_snapshot);
    impl_->gdn_state_parity = !impl_->gdn_state_parity;
    // Advance only the MTP K/V cache through the final accepted draft. This is
    // queued on the same stream as the next round, so no host synchronization
    // is needed here. The verifier has now produced the exact target hidden
    // state for the preceding draft, which is a better conditioner than the
    // draft model's approximate hidden state.
    status = impl_->launch_mtp_proposal(
        drafts.back(), nullptr, impl_->last_hidden - kHidden,
        impl_->mtp_position, nullptr);
    impl_->mtp_chain_active = false;
    if (!status.ok()) return status;
    ++impl_->mtp_position;
  }

  SpeculativeStepResult result;
  result.proposed_drafts = draft_count;
  result.accepted_drafts = accepted;
  result.tokens.reserve(accepted + 1);
  result.tokens.insert(result.tokens.end(), drafts.begin(), drafts.begin() + accepted);
  result.tokens.push_back(target_outputs[accepted]);
  return result;
}

std::size_t QwenDecodeSession::position() const noexcept {
  return impl_ == nullptr ? 0 : impl_->current_position;
}

const DecodeSessionSummary& QwenDecodeSession::summary() const noexcept {
  return impl_->summary;
}

}  // namespace gfxinfer
