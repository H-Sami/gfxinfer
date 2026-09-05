// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <time.h>

#include "gfxinfer/artifact.h"
#include "gfxinfer/converter.h"
#include "gfxinfer/device.h"
#include "gfxinfer/engine.h"
#include "gfxinfer/lowbit_gemv.h"
#include "gfxinfer/quantization.h"
#include "gfxinfer/qwen_ops.h"
#include "gfxinfer/qwen38_spec.h"
#include "gfxinfer/qwen_runtime.h"
#include "gfxinfer/safetensors.h"
#include "gfxinfer/tokenizer.h"
#include "gfxinfer/version.h"

namespace {

using gfxinfer::Quantization;

void print_usage() {
  std::cout
      << "GFXInfer " << gfxinfer::kVersion << "\n\n"
      << "Usage:\n"
      << "  gfxinfer probe\n"
      << "  gfxinfer inspect <artifact.gfxi>\n"
      << "  gfxinfer verify-source <Qwen3.8-27B-directory>\n"
      << "  gfxinfer inspect-safetensors <model-shard.safetensors>\n"
      << "  gfxinfer convert <Qwen3.8-27B-directory> <output.gfxi> [--plan-only] "
         "[--mlp-bits 2|3|4] [--head-bits 2|3|4] "
         "[--w3-start-layer N] [--w4-tail N] "
         "[--draft-ranking counts.i64]\n"
      << "  gfxinfer load-check <artifact.gfxi>\n"
      << "  gfxinfer model-smoke <artifact.gfxi> [--token N] [--iters N]\n"
      << "  gfxinfer gdn-smoke <artifact.gfxi> [--token N] [--iters N]\n"
      << "  gfxinfer attention-smoke <artifact.gfxi> [--token N] [--iters N]\n"
      << "  gfxinfer decode-bench <artifact.gfxi> [--token N] [--tokens N] [--context N]\n"
      << "  gfxinfer block-bench <artifact.gfxi> [--token N] [--batch 1..32] "
         "[--blocks N] [--context N] [--activation f16|a8|a4|a4w2|a4w4]\n"
      << "  gfxinfer spec-bench <artifact.gfxi> [--token N] [--drafts 1..31] "
         "[--tokens N] [--context N] [--activation f16|a8|a4|a4w2|a4w4]\n"
      << "  gfxinfer tokenizer-check <artifact.gfxi> <text>\n"
      << "  gfxinfer generate <artifact.gfxi> --prompt <text> [--max-tokens N] "
         "[--context N] [--activation f16|a8|a4|a4w2|a4w4] [--drafts 0..31]\n"
      << "  gfxinfer mtp-bench <artifact.gfxi> [--token N] [--steps N] [--context N] "
         "[--activation f16|a8|a4|a4w2|a4w4]\n"
      << "  gfxinfer bench-gemv [--format w2|w3|w4] [--rows N] [--cols K] "
         "[--batch 1..32] [--kernel scalar|wmma-a4|wmma-a8|wmma-f16] "
         "[--iters N] [--device N]\n";
}

int fail(const gfxinfer::Status& status) {
  std::cerr << "error: " << status.message() << '\n';
  return 1;
}

double monotonic_milliseconds() {
  timespec value{};
  (void)clock_gettime(CLOCK_MONOTONIC_RAW, &value);
  return static_cast<double>(value.tv_sec) * 1000.0 +
         static_cast<double>(value.tv_nsec) / 1.0e6;
}

std::string gib(std::uint64_t bytes) {
  const double value = static_cast<double>(bytes) / static_cast<double>(1ULL << 30U);
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << value << " GiB";
  return output.str();
}

int probe() {
  auto devices = gfxinfer::enumerate_devices();
  if (!devices.ok()) return fail(devices.status());
  if (devices.value().empty()) {
    std::cout << "No HIP devices found.\n";
    return 1;
  }
  for (const auto& device : devices.value()) {
    std::cout << "device " << device.index << ": " << device.name << '\n'
              << "  architecture: " << device.architecture << '\n'
              << "  memory:       " << gib(device.total_memory_bytes) << '\n'
              << "  compute units:" << ' ' << device.compute_units << '\n'
              << "  wave size:    " << device.warp_size << '\n'
              << "  core clock:   " << device.clock_khz / 1000 << " MHz\n"
              << "  memory clock: " << device.memory_clock_khz / 1000 << " MHz\n"
              << "  memory bus:   " << device.memory_bus_width_bits << " bit\n"
              << "  eligible:     "
              << (device.architecture == gfxinfer::kRequiredArchitecture && device.warp_size == 32
                      ? "yes"
                      : "no")
              << '\n';
  }
  return 0;
}

int inspect(const std::string& path) {
  auto artifact = gfxinfer::ArtifactView::open(path);
  if (!artifact.ok()) return fail(artifact.status());
  const auto& identity = artifact.value().identity();
  std::cout << "artifact: " << path << '\n'
            << "bytes:    " << artifact.value().file_bytes() << " ("
            << gib(artifact.value().file_bytes()) << ")\n"
            << "target:   " << identity.target_arch << '\n'
            << "model:    " << identity.model_id << '\n'
            << "weights:  " << identity.weights_id << '\n'
            << "recipe:   " << identity.recipe_id << '\n'
            << "tensors:  " << artifact.value().tensors().size() << "\n\n";
  for (const auto& tensor : artifact.value().tensors()) {
    std::cout << tensor.name << "  " << gfxinfer::dtype_name(tensor.dtype) << "  "
              << gfxinfer::quantization_name(tensor.quantization) << "  [";
    for (std::uint16_t dimension = 0; dimension < tensor.rank; ++dimension) {
      if (dimension != 0) std::cout << ',';
      std::cout << tensor.dimensions[dimension];
    }
    std::cout << "]  data=" << tensor.data_bytes << " aux=" << tensor.aux_bytes << '\n';
  }
  return 0;
}

int verify_source(const std::string& path) {
  auto summary = gfxinfer::validate_qwen38_model_directory(path, true, true);
  if (!summary.ok()) return fail(summary.status());
  std::cout << "Qwen3.8-27B source validation passed\n"
            << "  directory:    " << path << '\n'
            << "  shards:       " << summary.value().shard_count << '\n'
            << "  tensors:      " << summary.value().tensor_count << '\n'
            << "  tensor bytes: " << summary.value().tensor_bytes << " ("
            << gib(summary.value().tensor_bytes) << ")\n";
  return 0;
}

int inspect_safetensors(const std::string& path) {
  auto file = gfxinfer::SafeTensorFile::open(path);
  if (!file.ok()) return fail(file.status());
  std::cout << "shard:   " << path << '\n'
            << "bytes:   " << file.value().file_bytes() << '\n'
            << "tensors: " << file.value().tensors().size() << "\n\n";
  for (const auto& tensor : file.value().tensors()) {
    std::cout << tensor.name << "  " << gfxinfer::source_dtype_name(tensor.dtype) << "  [";
    for (std::size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
      if (dimension != 0) std::cout << ',';
      std::cout << tensor.shape[dimension];
    }
    std::cout << "]  bytes=" << tensor.bytes << '\n';
  }
  return 0;
}

int convert(int argc, char** argv) {
  if (argc < 4 || argc > 15) {
    print_usage();
    return 1;
  }
  gfxinfer::Qwen38ConversionOptions options;
  options.source_directory = argv[2];
  options.output_path = argv[3];
  for (int index = 4; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--plan-only") {
      options.plan_only = true;
      continue;
    }
    if (index + 1 >= argc) {
      std::cerr << "error: conversion option requires a value: " << option << '\n';
      return 1;
    }
    const std::string_view value = argv[++index];
    if (option == "--draft-ranking") {
      options.draft_ranking_path = value;
      continue;
    }
    std::size_t parsed = 0;
    try {
      std::size_t consumed = 0;
      parsed = static_cast<std::size_t>(std::stoull(std::string(value), &consumed));
      if (consumed != value.size()) throw std::invalid_argument("trailing characters");
    } catch (...) {
      std::cerr << "error: invalid numeric conversion option: " << option << '\n';
      return 1;
    }
    if (option == "--mlp-bits" &&
        (parsed == 2 || parsed == 3 || parsed == 4)) {
      options.mlp_bits = static_cast<unsigned>(parsed);
    } else if (option == "--head-bits" &&
               (parsed == 2 || parsed == 3 || parsed == 4)) {
      options.output_head_bits = static_cast<unsigned>(parsed);
    } else if (option == "--w3-start-layer") {
      options.w3_start_layer = parsed;
    } else if (option == "--w4-tail") {
      options.w4_tail_layers = parsed;
    } else {
      std::cerr << "error: invalid conversion option: " << option << '\n';
      return 1;
    }
  }
  std::size_t last_report = std::numeric_limits<std::size_t>::max();
  auto summary = gfxinfer::convert_qwen38(
      options, [&](std::size_t complete, std::size_t total, std::string_view name) {
        const auto bucket = total == 0 ? 0 : complete * 100 / total;
        if (complete == total || bucket / 2 != last_report / 2) {
          std::cout << '[' << std::setw(3) << bucket << "%] " << name << '\n' << std::flush;
          last_report = bucket;
        }
      });
  if (!summary.ok()) return fail(summary.status());
  std::cout << (options.plan_only ? "conversion plan ready\n" : "conversion complete\n")
            << "  model tensors: " << summary.value().model_tensor_count << '\n'
            << "  resources:     " << summary.value().resource_count << '\n'
            << "  W2 matrices:   " << summary.value().w2_tensor_count << '\n'
            << "  W3 matrices:   " << summary.value().w3_tensor_count << '\n'
            << "  W4 matrices:   " << summary.value().w4_tensor_count << '\n'
            << "  BF16 tensors:  " << summary.value().bf16_tensor_count << '\n'
            << "  payload:       " << gib(summary.value().planned_payload_bytes) << '\n';
  if (!options.plan_only) {
    std::cout << "  artifact:      " << options.output_path << '\n'
              << "  file size:     " << gib(summary.value().output_file_bytes) << '\n';
  }
  return 0;
}

int load_check(const std::string& path) {
  gfxinfer::EngineConfig config;
  config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(config));
  if (!engine.ok()) return fail(engine.status());
  const auto& summary = engine.value().load_summary();
  std::cout << "resident load passed\n"
            << "  device:         " << summary.device.name << " ("
            << summary.device.architecture << ")\n"
            << "  artifact:       " << gib(summary.artifact_bytes) << '\n'
            << "  tensors:        " << summary.resident_tensor_count << " resident / "
            << summary.tensor_count << " total\n"
            << "  device arena:   " << gib(summary.resident_weight_bytes) << '\n'
            << "  free before:    " << gib(summary.device_free_bytes_before_load) << '\n'
            << "  free after:     " << gib(summary.device_free_bytes_after_load) << '\n'
            << "  memory budget:  " << gib(summary.configured_memory_budget_bytes) << '\n';
  return 0;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) (void)hipFree(pointer_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  gfxinfer::Status allocate(std::size_t count) {
    count_ = count;
    const auto error = hipMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T));
    return gfxinfer::hip_status(error, "hipMalloc failed");
  }
  T* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

 private:
  T* pointer_{nullptr};
  std::size_t count_{0};
};

gfxinfer::Status launch_matrix(
    const gfxinfer::DeviceTensorView& tensor,
    const __half* input,
    float* output,
    hipStream_t stream) {
  if (tensor.rank != 2 || tensor.data == nullptr || tensor.auxiliary == nullptr) {
    return {gfxinfer::ErrorCode::invalid_argument,
            "device tensor is not a resident quantized matrix: " + tensor.name};
  }
  return gfxinfer::launch_lowbit_gemv(
      tensor.quantization,
      reinterpret_cast<const std::uint8_t*>(tensor.data),
      reinterpret_cast<const __half*>(tensor.auxiliary),
      input,
      output,
      static_cast<std::size_t>(tensor.dimensions[0]),
      static_cast<std::size_t>(tensor.dimensions[1]),
      stream);
}

int model_smoke(const std::string& path, std::size_t token, int iterations) {
  gfxinfer::EngineConfig config;
  config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(config));
  if (!engine.ok()) return fail(engine.status());
  auto& runtime = engine.value();
  const auto* embedding = runtime.find_device_tensor(
      "model.language_model.embed_tokens.weight");
  const auto* norm = runtime.find_device_tensor(
      "model.language_model.layers.0.post_attention_layernorm.weight");
  const auto* gate = runtime.find_device_tensor(
      "model.language_model.layers.0.mlp.gate_proj.weight");
  const auto* up = runtime.find_device_tensor(
      "model.language_model.layers.0.mlp.up_proj.weight");
  const auto* down = runtime.find_device_tensor(
      "model.language_model.layers.0.mlp.down_proj.weight");
  if (embedding == nullptr || norm == nullptr || gate == nullptr || up == nullptr ||
      down == nullptr || token >= embedding->dimensions[0]) {
    std::cerr << "error: required layer-0 tensors or token row are unavailable\n";
    return 1;
  }

  constexpr std::size_t hidden = gfxinfer::Qwen38Spec::hidden_size;
  constexpr std::size_t intermediate = gfxinfer::Qwen38Spec::intermediate_size;
  DeviceBuffer<__half> hidden_state;
  DeviceBuffer<__half> normalized;
  DeviceBuffer<__half> activated;
  DeviceBuffer<__half> layer_output;
  DeviceBuffer<float> gate_output;
  DeviceBuffer<float> up_output;
  DeviceBuffer<float> down_output;
  for (auto status : {
           hidden_state.allocate(hidden), normalized.allocate(hidden),
           activated.allocate(intermediate), layer_output.allocate(hidden),
           gate_output.allocate(intermediate), up_output.allocate(intermediate),
           down_output.allocate(hidden)}) {
    if (!status.ok()) return fail(status);
  }
  auto stream = runtime.compute_stream();
  auto status = gfxinfer::launch_lowbit_row_gather(
      embedding->quantization,
      reinterpret_cast<const std::uint8_t*>(embedding->data),
      reinterpret_cast<const __half*>(embedding->auxiliary),
      hidden_state.get(), token, static_cast<std::size_t>(embedding->dimensions[0]), hidden,
      stream);
  if (!status.ok()) return fail(status);
  status = gfxinfer::launch_rms_norm(
      hidden_state.get(), reinterpret_cast<const std::uint16_t*>(norm->data),
      normalized.get(), hidden, gfxinfer::Qwen38Spec::rms_norm_epsilon, true, stream);
  if (!status.ok()) return fail(status);

  auto launch_mlp = [&]() -> gfxinfer::Status {
    auto current = launch_matrix(*gate, normalized.get(), gate_output.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*up, normalized.get(), up_output.get(), stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_silu_multiply(
        gate_output.get(), up_output.get(), activated.get(), intermediate, stream);
    if (!current.ok()) return current;
    current = launch_matrix(*down, activated.get(), down_output.get(), stream);
    if (!current.ok()) return current;
    return gfxinfer::launch_residual_add(
        hidden_state.get(), down_output.get(), layer_output.get(), hidden, stream);
  };
  for (int warmup = 0; warmup < 5; ++warmup) {
    status = launch_mlp();
    if (!status.ok()) return fail(status);
  }
  if (auto error = hipStreamSynchronize(stream); error != hipSuccess) {
    return fail(gfxinfer::hip_status(error, "model smoke warmup failed"));
  }

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (auto error = hipEventCreate(&start); error != hipSuccess)
    return fail(gfxinfer::hip_status(error, "model smoke event creation failed"));
  if (auto error = hipEventCreate(&stop); error != hipSuccess) {
    (void)hipEventDestroy(start);
    return fail(gfxinfer::hip_status(error, "model smoke event creation failed"));
  }
  (void)hipEventRecord(start, stream);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    status = launch_mlp();
    if (!status.ok()) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(stop);
      return fail(status);
    }
  }
  (void)hipEventRecord(stop, stream);
  if (auto error = hipEventSynchronize(stop); error != hipSuccess) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(gfxinfer::hip_status(error, "model smoke timing failed"));
  }
  float elapsed_ms = 0.0F;
  (void)hipEventElapsedTime(&elapsed_ms, start, stop);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);

  std::vector<__half> host_output(hidden);
  if (auto error = hipMemcpy(host_output.data(), layer_output.get(), layer_output.bytes(),
                             hipMemcpyDeviceToHost);
      error != hipSuccess) {
    return fail(gfxinfer::hip_status(error, "model smoke output download failed"));
  }
  double square_sum = 0.0;
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  for (const auto value_half : host_output) {
    const float value = __half2float(value_half);
    if (!std::isfinite(value)) {
      std::cerr << "error: layer-0 MLP produced a non-finite value\n";
      return 1;
    }
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    square_sum += static_cast<double>(value) * value;
  }
  const float mean_ms = elapsed_ms / static_cast<float>(iterations);
  std::cout << std::fixed << std::setprecision(4)
            << "real-model GPU smoke passed\n"
            << "  token row:       " << token << '\n'
            << "  path:            embedding -> RMSNorm -> W2 MLP -> residual\n"
            << "  mean MLP time:   " << mean_ms << " ms\n"
            << "  MLPs/second:     " << 1000.0F / mean_ms << '\n'
            << "  output range:    [" << minimum << ", " << maximum << "]\n"
            << "  output RMS:      " << std::sqrt(square_sum / hidden) << '\n'
            << "  validation:      finite\n";
  return 0;
}

int gdn_smoke(const std::string& path, std::size_t token, int iterations) {
  gfxinfer::EngineConfig config;
  config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(config));
  if (!engine.ok()) return fail(engine.status());
  auto& runtime = engine.value();
  auto tensor = [&](std::string_view name) { return runtime.find_device_tensor(name); };
  const auto* embedding = tensor("model.language_model.embed_tokens.weight");
  const auto* input_norm = tensor("model.language_model.layers.0.input_layernorm.weight");
  const auto* qkv = tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
  const auto* z = tensor("model.language_model.layers.0.linear_attn.in_proj_z.weight");
  const auto* a = tensor("model.language_model.layers.0.linear_attn.in_proj_a.weight");
  const auto* b = tensor("model.language_model.layers.0.linear_attn.in_proj_b.weight");
  const auto* convolution = tensor("model.language_model.layers.0.linear_attn.conv1d.weight");
  const auto* a_log = tensor("model.language_model.layers.0.linear_attn.A_log");
  const auto* dt_bias = tensor("model.language_model.layers.0.linear_attn.dt_bias");
  const auto* gdn_norm = tensor("model.language_model.layers.0.linear_attn.norm.weight");
  const auto* out = tensor("model.language_model.layers.0.linear_attn.out_proj.weight");
  const auto* post_norm = tensor(
      "model.language_model.layers.0.post_attention_layernorm.weight");
  const auto* gate = tensor("model.language_model.layers.0.mlp.gate_proj.weight");
  const auto* up = tensor("model.language_model.layers.0.mlp.up_proj.weight");
  const auto* down = tensor("model.language_model.layers.0.mlp.down_proj.weight");
  const std::array required{embedding, input_norm, qkv, z, a, b, convolution, a_log,
                            dt_bias, gdn_norm, out, post_norm, gate, up, down};
  if (std::any_of(required.begin(), required.end(), [](const auto* value) {
        return value == nullptr;
      }) || token >= embedding->dimensions[0]) {
    std::cerr << "error: required layer-0 GDN tensors or token row are unavailable\n";
    return 1;
  }

  constexpr std::size_t hidden = gfxinfer::Qwen38Spec::hidden_size;
  constexpr std::size_t intermediate = gfxinfer::Qwen38Spec::intermediate_size;
  constexpr std::size_t qkv_elements = 10240;
  constexpr std::size_t value_elements = 6144;
  constexpr std::size_t heads = 48;
  constexpr std::size_t head_size = 128;
  DeviceBuffer<__half> hidden_state, normalized, convolved_qkv, conv_state;
  DeviceBuffer<__half> gdn_output, post_attention, post_normalized, activated, final_output;
  DeviceBuffer<float> projected_qkv, projected_z, projected_a, projected_b;
  DeviceBuffer<float> recurrent_state, attention_update, gate_output, up_output, mlp_update;
  for (auto status : {
           hidden_state.allocate(hidden), normalized.allocate(hidden),
           convolved_qkv.allocate(qkv_elements), conv_state.allocate(qkv_elements * 4),
           gdn_output.allocate(value_elements), post_attention.allocate(hidden),
           post_normalized.allocate(hidden), activated.allocate(intermediate),
           final_output.allocate(hidden), projected_qkv.allocate(qkv_elements),
           projected_z.allocate(value_elements), projected_a.allocate(heads),
           projected_b.allocate(heads),
           recurrent_state.allocate(heads * head_size * head_size),
           attention_update.allocate(hidden), gate_output.allocate(intermediate),
           up_output.allocate(intermediate), mlp_update.allocate(hidden)}) {
    if (!status.ok()) return fail(status);
  }
  auto stream = runtime.compute_stream();
  if (auto error = hipMemsetAsync(conv_state.get(), 0, conv_state.bytes(), stream);
      error != hipSuccess) {
    return fail(gfxinfer::hip_status(error, "GDN convolution state initialization failed"));
  }
  if (auto error = hipMemsetAsync(recurrent_state.get(), 0, recurrent_state.bytes(), stream);
      error != hipSuccess) {
    return fail(gfxinfer::hip_status(error, "GDN recurrent state initialization failed"));
  }
  auto status = gfxinfer::launch_lowbit_row_gather(
      embedding->quantization,
      reinterpret_cast<const std::uint8_t*>(embedding->data),
      reinterpret_cast<const __half*>(embedding->auxiliary), hidden_state.get(), token,
      static_cast<std::size_t>(embedding->dimensions[0]), hidden, stream);
  if (!status.ok()) return fail(status);

  auto launch_layer = [&]() -> gfxinfer::Status {
    auto current = gfxinfer::launch_rms_norm(
        hidden_state.get(), reinterpret_cast<const std::uint16_t*>(input_norm->data),
        normalized.get(), hidden, gfxinfer::Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!current.ok()) return current;
    current = launch_matrix(*qkv, normalized.get(), projected_qkv.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*z, normalized.get(), projected_z.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*a, normalized.get(), projected_a.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*b, normalized.get(), projected_b.get(), stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_gdn_conv_update(
        projected_qkv.get(), reinterpret_cast<const std::uint16_t*>(convolution->data),
        conv_state.get(), convolved_qkv.get(), stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_gdn_recurrent_decode(
        convolved_qkv.get(), projected_z.get(), projected_a.get(), projected_b.get(),
        reinterpret_cast<const std::uint16_t*>(a_log->data),
        reinterpret_cast<const std::uint16_t*>(dt_bias->data),
        reinterpret_cast<const std::uint16_t*>(gdn_norm->data), recurrent_state.get(),
        gdn_output.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*out, gdn_output.get(), attention_update.get(), stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_residual_add(
        hidden_state.get(), attention_update.get(), post_attention.get(), hidden, stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_rms_norm(
        post_attention.get(), reinterpret_cast<const std::uint16_t*>(post_norm->data),
        post_normalized.get(), hidden, gfxinfer::Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!current.ok()) return current;
    current = launch_matrix(*gate, post_normalized.get(), gate_output.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*up, post_normalized.get(), up_output.get(), stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_silu_multiply(
        gate_output.get(), up_output.get(), activated.get(), intermediate, stream);
    if (!current.ok()) return current;
    current = launch_matrix(*down, activated.get(), mlp_update.get(), stream);
    if (!current.ok()) return current;
    return gfxinfer::launch_residual_add(
        post_attention.get(), mlp_update.get(), final_output.get(), hidden, stream);
  };
  for (int warmup = 0; warmup < 5; ++warmup) {
    status = launch_layer();
    if (!status.ok()) return fail(status);
  }
  if (auto error = hipStreamSynchronize(stream); error != hipSuccess)
    return fail(gfxinfer::hip_status(error, "GDN layer warmup failed"));

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (auto error = hipEventCreate(&start); error != hipSuccess)
    return fail(gfxinfer::hip_status(error, "GDN smoke event creation failed"));
  if (auto error = hipEventCreate(&stop); error != hipSuccess) {
    (void)hipEventDestroy(start);
    return fail(gfxinfer::hip_status(error, "GDN smoke event creation failed"));
  }
  (void)hipEventRecord(start, stream);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    status = launch_layer();
    if (!status.ok()) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(stop);
      return fail(status);
    }
  }
  (void)hipEventRecord(stop, stream);
  if (auto error = hipEventSynchronize(stop); error != hipSuccess) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(gfxinfer::hip_status(error, "GDN layer timing failed"));
  }
  float elapsed_ms = 0.0F;
  (void)hipEventElapsedTime(&elapsed_ms, start, stop);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);

  std::vector<__half> host_output(hidden);
  if (auto error = hipMemcpy(host_output.data(), final_output.get(), final_output.bytes(),
                             hipMemcpyDeviceToHost);
      error != hipSuccess) {
    return fail(gfxinfer::hip_status(error, "GDN layer output download failed"));
  }
  double square_sum = 0.0;
  float maximum_absolute = 0.0F;
  for (const auto value_half : host_output) {
    const float value = __half2float(value_half);
    if (!std::isfinite(value)) {
      std::cerr << "error: full layer produced a non-finite value\n";
      return 1;
    }
    maximum_absolute = std::max(maximum_absolute, std::abs(value));
    square_sum += static_cast<double>(value) * value;
  }
  const float mean_ms = elapsed_ms / static_cast<float>(iterations);
  std::cout << std::fixed << std::setprecision(4)
            << "real GDN decoder-layer smoke passed\n"
            << "  token row:       " << token << '\n'
            << "  path:            RMSNorm -> GDN -> residual -> RMSNorm -> MLP -> residual\n"
            << "  mean layer time: " << mean_ms << " ms\n"
            << "  layers/second:   " << 1000.0F / mean_ms << '\n'
            << "  output max abs:  " << maximum_absolute << '\n'
            << "  output RMS:      " << std::sqrt(square_sum / hidden) << '\n'
            << "  state bytes:     " << recurrent_state.bytes() + conv_state.bytes() << '\n'
            << "  validation:      finite\n";
  return 0;
}

int attention_smoke(const std::string& path, std::size_t token, int iterations) {
  gfxinfer::EngineConfig config;
  config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(config));
  if (!engine.ok()) return fail(engine.status());
  auto& runtime = engine.value();
  auto tensor = [&](std::string_view name) { return runtime.find_device_tensor(name); };
  constexpr std::string_view prefix = "model.language_model.layers.3.";
  const auto* embedding = tensor("model.language_model.embed_tokens.weight");
  const auto* input_norm = tensor(std::string(prefix) + "input_layernorm.weight");
  const auto* q = tensor(std::string(prefix) + "self_attn.q_proj.weight");
  const auto* k = tensor(std::string(prefix) + "self_attn.k_proj.weight");
  const auto* v = tensor(std::string(prefix) + "self_attn.v_proj.weight");
  const auto* q_norm = tensor(std::string(prefix) + "self_attn.q_norm.weight");
  const auto* k_norm = tensor(std::string(prefix) + "self_attn.k_norm.weight");
  const auto* o = tensor(std::string(prefix) + "self_attn.o_proj.weight");
  const auto* post_norm = tensor(std::string(prefix) + "post_attention_layernorm.weight");
  const auto* gate = tensor(std::string(prefix) + "mlp.gate_proj.weight");
  const auto* up = tensor(std::string(prefix) + "mlp.up_proj.weight");
  const auto* down = tensor(std::string(prefix) + "mlp.down_proj.weight");
  const std::array required{embedding, input_norm, q, k, v, q_norm,
                            k_norm, o, post_norm, gate, up, down};
  if (std::any_of(required.begin(), required.end(), [](const auto* value) {
        return value == nullptr;
      }) || token >= embedding->dimensions[0]) {
    std::cerr << "error: required layer-3 attention tensors or token row are unavailable\n";
    return 1;
  }

  constexpr std::size_t hidden = gfxinfer::Qwen38Spec::hidden_size;
  constexpr std::size_t intermediate = gfxinfer::Qwen38Spec::intermediate_size;
  constexpr std::size_t query_elements = 24 * 256;
  constexpr std::size_t q_projection_elements = query_elements * 2;
  constexpr std::size_t kv_elements = 4 * 256;
  constexpr std::size_t warmups = 5;
  const auto capacity = warmups + static_cast<std::size_t>(iterations);
  DeviceBuffer<__half> hidden_state, normalized, query, key, attention_output;
  DeviceBuffer<__half> key_cache, value_cache, post_attention, post_normalized;
  DeviceBuffer<__half> activated, final_output;
  DeviceBuffer<float> projected_q, projected_k, projected_v, attention_update;
  DeviceBuffer<float> gate_output, up_output, mlp_update;
  for (auto status : {
           hidden_state.allocate(hidden), normalized.allocate(hidden),
           query.allocate(query_elements), key.allocate(kv_elements),
           attention_output.allocate(query_elements), key_cache.allocate(capacity * kv_elements),
           value_cache.allocate(capacity * kv_elements), post_attention.allocate(hidden),
           post_normalized.allocate(hidden), activated.allocate(intermediate),
           final_output.allocate(hidden), projected_q.allocate(q_projection_elements),
           projected_k.allocate(kv_elements), projected_v.allocate(kv_elements),
           attention_update.allocate(hidden), gate_output.allocate(intermediate),
           up_output.allocate(intermediate), mlp_update.allocate(hidden)}) {
    if (!status.ok()) return fail(status);
  }
  auto stream = runtime.compute_stream();
  auto status = gfxinfer::launch_lowbit_row_gather(
      embedding->quantization,
      reinterpret_cast<const std::uint8_t*>(embedding->data),
      reinterpret_cast<const __half*>(embedding->auxiliary), hidden_state.get(), token,
      static_cast<std::size_t>(embedding->dimensions[0]), hidden, stream);
  if (!status.ok()) return fail(status);

  auto launch_layer = [&](std::size_t position) -> gfxinfer::Status {
    auto current = gfxinfer::launch_rms_norm(
        hidden_state.get(), reinterpret_cast<const std::uint16_t*>(input_norm->data),
        normalized.get(), hidden, gfxinfer::Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!current.ok()) return current;
    current = launch_matrix(*q, normalized.get(), projected_q.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*k, normalized.get(), projected_k.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*v, normalized.get(), projected_v.get(), stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_head_rms_norm_rope(
        projected_q.get(), reinterpret_cast<const std::uint16_t*>(q_norm->data),
        query.get(), 24, 256, 512, 64, position, gfxinfer::Qwen38Spec::rope_theta, stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_head_rms_norm_rope(
        projected_k.get(), reinterpret_cast<const std::uint16_t*>(k_norm->data),
        key.get(), 4, 256, 256, 64, position, gfxinfer::Qwen38Spec::rope_theta, stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_kv_append(
        key.get(), projected_v.get(), key_cache.get(), value_cache.get(), position,
        capacity, stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_gated_attention_decode(
        query.get(), projected_q.get(), key_cache.get(), value_cache.get(),
        attention_output.get(), position + 1, capacity, stream);
    if (!current.ok()) return current;
    current = launch_matrix(*o, attention_output.get(), attention_update.get(), stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_residual_add(
        hidden_state.get(), attention_update.get(), post_attention.get(), hidden, stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_rms_norm(
        post_attention.get(), reinterpret_cast<const std::uint16_t*>(post_norm->data),
        post_normalized.get(), hidden, gfxinfer::Qwen38Spec::rms_norm_epsilon, true, stream);
    if (!current.ok()) return current;
    current = launch_matrix(*gate, post_normalized.get(), gate_output.get(), stream);
    if (!current.ok()) return current;
    current = launch_matrix(*up, post_normalized.get(), up_output.get(), stream);
    if (!current.ok()) return current;
    current = gfxinfer::launch_silu_multiply(
        gate_output.get(), up_output.get(), activated.get(), intermediate, stream);
    if (!current.ok()) return current;
    current = launch_matrix(*down, activated.get(), mlp_update.get(), stream);
    if (!current.ok()) return current;
    return gfxinfer::launch_residual_add(
        post_attention.get(), mlp_update.get(), final_output.get(), hidden, stream);
  };
  for (std::size_t position = 0; position < warmups; ++position) {
    status = launch_layer(position);
    if (!status.ok()) return fail(status);
  }
  if (auto error = hipStreamSynchronize(stream); error != hipSuccess)
    return fail(gfxinfer::hip_status(error, "attention layer warmup failed"));

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (auto error = hipEventCreate(&start); error != hipSuccess)
    return fail(gfxinfer::hip_status(error, "attention smoke event creation failed"));
  if (auto error = hipEventCreate(&stop); error != hipSuccess) {
    (void)hipEventDestroy(start);
    return fail(gfxinfer::hip_status(error, "attention smoke event creation failed"));
  }
  (void)hipEventRecord(start, stream);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    status = launch_layer(warmups + static_cast<std::size_t>(iteration));
    if (!status.ok()) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(stop);
      return fail(status);
    }
  }
  (void)hipEventRecord(stop, stream);
  if (auto error = hipEventSynchronize(stop); error != hipSuccess) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(gfxinfer::hip_status(error, "attention layer timing failed"));
  }
  float elapsed_ms = 0.0F;
  (void)hipEventElapsedTime(&elapsed_ms, start, stop);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);

  std::vector<__half> host_output(hidden);
  if (auto error = hipMemcpy(host_output.data(), final_output.get(), final_output.bytes(),
                             hipMemcpyDeviceToHost);
      error != hipSuccess) {
    return fail(gfxinfer::hip_status(error, "attention layer output download failed"));
  }
  double square_sum = 0.0;
  float maximum_absolute = 0.0F;
  for (const auto value_half : host_output) {
    const float value = __half2float(value_half);
    if (!std::isfinite(value)) {
      std::cerr << "error: attention layer produced a non-finite value\n";
      return 1;
    }
    maximum_absolute = std::max(maximum_absolute, std::abs(value));
    square_sum += static_cast<double>(value) * value;
  }
  const float mean_ms = elapsed_ms / static_cast<float>(iterations);
  std::cout << std::fixed << std::setprecision(4)
            << "real full-attention decoder-layer smoke passed\n"
            << "  token row:       " << token << '\n'
            << "  timed contexts:  " << warmups + 1 << ".." << capacity << " tokens\n"
            << "  path:            RMSNorm -> GQA/RoPE/KV -> residual -> MLP -> residual\n"
            << "  mean layer time: " << mean_ms << " ms\n"
            << "  layers/second:   " << 1000.0F / mean_ms << '\n'
            << "  output max abs:  " << maximum_absolute << '\n'
            << "  output RMS:      " << std::sqrt(square_sum / hidden) << '\n'
            << "  KV bytes:        " << key_cache.bytes() + value_cache.bytes() << '\n'
            << "  validation:      finite\n";
  return 0;
}

int decode_bench(
    const std::string& path,
    std::size_t starting_token,
    std::size_t token_count,
    std::size_t context) {
  gfxinfer::EngineConfig engine_config;
  engine_config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(engine_config));
  if (!engine.ok()) return fail(engine.status());
  auto tokenizer = gfxinfer::QwenTokenizer::from_json(
      engine.value().resource("resources/tokenizer.json"));
  if (!tokenizer.ok()) return fail(tokenizer.status());
  auto session = gfxinfer::QwenDecodeSession::create(
      engine.value(), {.maximum_context_tokens = context});
  if (!session.ok()) return fail(session.status());
  std::vector<std::uint32_t> output_tokens;
  std::vector<double> durations_ms;
  output_tokens.reserve(token_count);
  durations_ms.reserve(token_count);
  auto input_token = static_cast<std::uint32_t>(starting_token);
  for (std::size_t index = 0; index < token_count; ++index) {
    const double begin = monotonic_milliseconds();
    auto output = session.value().greedy_step(input_token);
    const double end = monotonic_milliseconds();
    if (!output.ok()) return fail(output.status());
    input_token = output.value();
    output_tokens.push_back(input_token);
    durations_ms.push_back(end - begin);
  }
  const double total_ms = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);
  auto ordered = durations_ms;
  std::sort(ordered.begin(), ordered.end());
  auto decoded = tokenizer.value().decode(output_tokens);
  if (!decoded.ok()) return fail(decoded.status());
  const auto& summary = session.value().summary();
  std::cout << std::fixed << std::setprecision(3)
            << "complete 64-layer greedy decode passed\n"
            << "  generated tokens: " << token_count << '\n'
            << "  total time:       " << total_ms << " ms\n"
            << "  decode speed:     " << token_count * 1000.0 / total_ms << " tokens/s\n"
            << "  mean latency:     " << total_ms / token_count << " ms/token\n"
            << "  p50 latency:      " << ordered[ordered.size() / 2] << " ms\n"
            << "  session arena:    " << gib(summary.arena_bytes) << '\n'
            << "  GDN state:        " << gib(summary.gdn_state_bytes) << '\n'
            << "  KV cache:         " << gib(summary.kv_cache_bytes) << " for "
            << summary.maximum_context_tokens << " tokens\n"
            << "  output token IDs: ";
  for (std::size_t index = 0; index < output_tokens.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << output_tokens[index];
  }
  std::cout << "\n  decoded bytes:    " << std::quoted(decoded.value())
            << "\n  validation:       all kernels completed\n";
  return 0;
}

int block_bench(
    const std::string& path,
    std::size_t starting_token,
    std::size_t batch,
    std::size_t block_count,
    std::size_t context,
    gfxinfer::ActivationMode activation_mode) {
  gfxinfer::EngineConfig engine_config;
  engine_config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(engine_config));
  if (!engine.ok()) return fail(engine.status());
  auto serial = gfxinfer::QwenDecodeSession::create(
      engine.value(), {.maximum_context_tokens = context,
                       .activation_mode = activation_mode});
  if (!serial.ok()) return fail(serial.status());
  auto blocked = gfxinfer::QwenDecodeSession::create(
      engine.value(), {.maximum_context_tokens = context,
                       .activation_mode = activation_mode});
  if (!blocked.ok()) return fail(blocked.status());

  std::vector<std::uint32_t> inputs(batch);
  for (std::size_t index = 0; index < batch; ++index) {
    inputs[index] = static_cast<std::uint32_t>(
        (starting_token + index) % gfxinfer::Qwen38Spec::tokenizer_vocabulary_size);
  }
  std::vector<std::uint32_t> serial_outputs;
  serial_outputs.reserve(batch);
  for (const auto token : inputs) {
    auto output = serial.value().greedy_step(token);
    if (!output.ok()) return fail(output.status());
    serial_outputs.push_back(output.value());
  }
  auto block_outputs = blocked.value().block_step(inputs);
  if (!block_outputs.ok()) return fail(block_outputs.status());
  if (block_outputs.value() != serial_outputs) {
    std::cerr << "error: causal block output differs from serial decode\n  serial: ";
    for (const auto token : serial_outputs) std::cerr << token << ' ';
    std::cerr << "\n  block:  ";
    for (const auto token : block_outputs.value()) std::cerr << token << ' ';
    std::cerr << '\n';
    return 1;
  }

  inputs = block_outputs.value();
  std::vector<double> durations;
  durations.reserve(block_count);
  for (std::size_t block = 0; block < block_count; ++block) {
    const double begin = monotonic_milliseconds();
    auto output = blocked.value().block_step(inputs);
    const double end = monotonic_milliseconds();
    if (!output.ok()) return fail(output.status());
    durations.push_back(end - begin);
    inputs = std::move(output).value();
  }
  const double total_ms = std::accumulate(durations.begin(), durations.end(), 0.0);
  const auto tokens = block_count * batch;
  const auto& summary = blocked.value().summary();
  std::cout << std::fixed << std::setprecision(3)
            << "causal shared-weight block decode passed\n"
            << "  serial parity:   exact token IDs\n"
            << "  batch:           " << batch << " tokens\n"
            << "  measured blocks: " << block_count << '\n'
            << "  total tokens:    " << tokens << '\n'
            << "  total time:      " << total_ms << " ms\n"
            << "  block latency:   " << total_ms / block_count << " ms\n"
            << "  effective speed: " << tokens * 1000.0 / total_ms << " tokens/s\n"
            << "  session arena:   " << gib(summary.arena_bytes) << '\n'
            << "  validation:      64-layer causal state and logits match serial\n";
  return 0;
}

int speculative_bench(
    const std::string& path,
    std::size_t starting_token,
    std::size_t draft_count,
    std::size_t minimum_tokens,
    std::size_t context,
    gfxinfer::ActivationMode activation_mode) {
  gfxinfer::EngineConfig engine_config;
  engine_config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(engine_config));
  if (!engine.ok()) return fail(engine.status());
  auto speculative = gfxinfer::QwenDecodeSession::create(
      engine.value(), {.maximum_context_tokens = context,
                       .activation_mode = activation_mode,
                       .graph_verifier_batch =
                           static_cast<unsigned>(draft_count + 1)});
  if (!speculative.ok()) return fail(speculative.status());
  auto serial = gfxinfer::QwenDecodeSession::create(
      engine.value(), {.maximum_context_tokens = context,
                       .activation_mode = activation_mode});
  if (!serial.ok()) return fail(serial.status());

  auto first = speculative.value().greedy_step(
      static_cast<std::uint32_t>(starting_token));
  if (!first.ok()) return fail(first.status());
  std::vector<std::uint32_t> speculative_tokens{first.value()};
  std::size_t proposed = 0;
  std::size_t accepted = 0;
  std::size_t rounds = 0;
  auto seed = first.value();
  const double begin = monotonic_milliseconds();
  while (speculative_tokens.size() < minimum_tokens) {
    auto result = speculative.value().speculative_step(seed, draft_count);
    if (!result.ok()) return fail(result.status());
    proposed += result.value().proposed_drafts;
    accepted += result.value().accepted_drafts;
    ++rounds;
    speculative_tokens.insert(
        speculative_tokens.end(), result.value().tokens.begin(),
        result.value().tokens.end());
    seed = result.value().tokens.back();
  }
  const double elapsed_ms = monotonic_milliseconds() - begin;

  auto serial_next = serial.value().greedy_step(
      static_cast<std::uint32_t>(starting_token));
  if (!serial_next.ok()) return fail(serial_next.status());
  std::vector<std::uint32_t> serial_tokens{serial_next.value()};
  while (serial_tokens.size() < speculative_tokens.size()) {
    serial_next = serial.value().greedy_step(serial_next.value());
    if (!serial_next.ok()) return fail(serial_next.status());
    serial_tokens.push_back(serial_next.value());
  }
  if (serial_tokens != speculative_tokens) {
    auto mismatch = std::mismatch(
        serial_tokens.begin(), serial_tokens.end(), speculative_tokens.begin());
    const auto index = static_cast<std::size_t>(
        std::distance(serial_tokens.begin(), mismatch.first));
    std::cerr << "error: speculative stream differs from serial target at token "
              << index << "\n  serial: " << *mismatch.first
              << "\n  speculative: " << *mismatch.second << '\n';
    return 1;
  }

  const auto measured_tokens = speculative_tokens.size() - 1;
  std::cout << std::fixed << std::setprecision(3)
            << "transactional MTP speculative decode passed\n"
            << "  draft depth:      " << draft_count << '\n'
            << "  rounds:           " << rounds << '\n'
            << "  measured tokens:  " << measured_tokens << '\n'
            << "  proposed drafts:  " << proposed << '\n'
            << "  accepted drafts:  " << accepted << '\n'
            << "  acceptance:       "
            << (proposed == 0 ? 0.0 : accepted * 100.0 / proposed) << "%\n"
            << "  tokens/round:     "
            << (rounds == 0 ? 0.0 : static_cast<double>(measured_tokens) / rounds) << '\n'
            << "  total time:       " << elapsed_ms << " ms\n"
            << "  effective speed:  " << measured_tokens * 1000.0 / elapsed_ms
            << " tokens/s\n"
            << "  serial parity:    exact token IDs\n";
  return 0;
}

int tokenizer_check(const std::string& path, std::string_view text) {
  gfxinfer::EngineConfig config;
  config.artifact = path;
  config.verify_artifact_checksums = false;
  config.upload_weights = false;
  auto engine = gfxinfer::Engine::create(std::move(config));
  if (!engine.ok()) return fail(engine.status());
  auto tokenizer = gfxinfer::QwenTokenizer::from_json(
      engine.value().resource("resources/tokenizer.json"));
  if (!tokenizer.ok()) return fail(tokenizer.status());
  auto tokens = tokenizer.value().encode(text);
  if (!tokens.ok()) return fail(tokens.status());
  auto decoded = tokenizer.value().decode(tokens.value());
  if (!decoded.ok()) return fail(decoded.status());
  std::cout << "tokenizer check passed\n  IDs: ";
  for (std::size_t index = 0; index < tokens.value().size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << tokens.value()[index];
  }
  std::cout << "\n  decoded: " << std::quoted(decoded.value()) << '\n';
  return 0;
}

int generate(
    const std::string& path,
    std::string_view prompt,
    std::size_t maximum_tokens,
    std::size_t context,
    gfxinfer::ActivationMode activation_mode,
    std::size_t draft_count) {
  gfxinfer::EngineConfig engine_config;
  engine_config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(engine_config));
  if (!engine.ok()) return fail(engine.status());
  auto tokenizer = gfxinfer::QwenTokenizer::from_json(
      engine.value().resource("resources/tokenizer.json"));
  if (!tokenizer.ok()) return fail(tokenizer.status());
  const std::string rendered =
      "<|im_start|>user\n" + std::string(prompt) +
      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  auto prompt_tokens = tokenizer.value().encode(rendered);
  if (!prompt_tokens.ok()) return fail(prompt_tokens.status());
  if (prompt_tokens.value().empty() ||
      prompt_tokens.value().size() + maximum_tokens > context) {
    std::cerr << "error: rendered prompt and generation exceed the session context\n";
    return 1;
  }
  auto session = gfxinfer::QwenDecodeSession::create(
      engine.value(), {.maximum_context_tokens = context,
                       .activation_mode = activation_mode,
                       .graph_verifier_batch =
                           static_cast<unsigned>(draft_count == 0
                                                     ? 0
                                                     : draft_count + 1)});
  if (!session.ok()) return fail(session.status());

  const double prefill_begin = monotonic_milliseconds();
  std::uint32_t next = 0;
  auto result = session.value().greedy_step(prompt_tokens.value().front());
  if (!result.ok()) return fail(result.status());
  next = result.value();
  for (std::size_t index = 1; index < prompt_tokens.value().size(); ++index) {
    const auto token = prompt_tokens.value()[index];
    if (draft_count != 0) {
      auto sync = session.value().mtp_draft(token);
      if (!sync.ok()) return fail(sync.status());
    }
    auto result = session.value().greedy_step(token);
    if (!result.ok()) return fail(result.status());
    next = result.value();
  }
  const double first_token_ready = monotonic_milliseconds();
  std::vector<std::uint32_t> generated;
  generated.reserve(maximum_tokens);
  std::size_t proposed_drafts = 0;
  std::size_t accepted_drafts = 0;
  std::size_t speculative_rounds = 0;
  std::size_t speculative_emitted = 0;
  generated.push_back(next);
  while (generated.size() < maximum_tokens &&
         next != 248044U && next != 248046U) {
    if (draft_count == 0 || maximum_tokens - generated.size() == 1) {
      auto target = session.value().greedy_step(next);
      if (!target.ok()) return fail(target.status());
      next = target.value();
      generated.push_back(next);
      continue;
    }
    const auto round_depth = std::min<std::size_t>(
        draft_count, maximum_tokens - generated.size() - 1);
    auto speculative = session.value().speculative_step(next, round_depth);
    if (!speculative.ok()) return fail(speculative.status());
    proposed_drafts += speculative.value().proposed_drafts;
    accepted_drafts += speculative.value().accepted_drafts;
    ++speculative_rounds;
    bool terminal = false;
    for (const auto token : speculative.value().tokens) {
      if (generated.size() == maximum_tokens) break;
      generated.push_back(token);
      ++speculative_emitted;
      next = token;
      if (token == 248044U || token == 248046U) {
        terminal = true;
        break;
      }
    }
    if (terminal) break;
  }
  const double generation_end = monotonic_milliseconds();
  auto decoded = tokenizer.value().decode(generated);
  if (!decoded.ok()) return fail(decoded.status());
  const double generation_ms = generation_end - first_token_ready;
  std::cout << decoded.value() << "\n\n"
            << std::fixed << std::setprecision(3)
            << "prompt tokens: " << prompt_tokens.value().size() << '\n'
            << "output tokens: " << generated.size() << '\n'
            << "time to first token: " << first_token_ready - prefill_begin << " ms\n";
  if (generated.size() > 1 && generation_ms > 0.0) {
    std::cout << "decode speed: " << (generated.size() - 1) * 1000.0 / generation_ms
              << " tokens/s\n";
  }
  if (draft_count != 0) {
    std::cout << "decode policy: MTP depth " << draft_count << '\n'
              << "speculative rounds: " << speculative_rounds << '\n'
              << "MTP acceptance: "
              << (proposed_drafts == 0
                      ? 0.0
                      : accepted_drafts * 100.0 / proposed_drafts)
              << "% (" << accepted_drafts << '/' << proposed_drafts << ")\n"
              << "accepted tokens/round: "
              << (speculative_rounds == 0
                      ? 0.0
                      : 1.0 + static_cast<double>(accepted_drafts) /
                                  static_cast<double>(speculative_rounds))
              << '\n'
              << "speculative emitted: " << speculative_emitted << " tokens\n";
  }
  return 0;
}

int mtp_bench(
    const std::string& path,
    std::size_t starting_token,
    std::size_t steps,
    std::size_t context,
    gfxinfer::ActivationMode activation_mode) {
  gfxinfer::EngineConfig engine_config;
  engine_config.artifact = path;
  auto engine = gfxinfer::Engine::create(std::move(engine_config));
  if (!engine.ok()) return fail(engine.status());
  auto tokenizer = gfxinfer::QwenTokenizer::from_json(
      engine.value().resource("resources/tokenizer.json"));
  if (!tokenizer.ok()) return fail(tokenizer.status());
  auto session = gfxinfer::QwenDecodeSession::create(
      engine.value(), {.maximum_context_tokens = context,
                       .activation_mode = activation_mode});
  if (!session.ok()) return fail(session.status());

  double target_ms = 0.0;
  double draft_ms = 0.0;
  std::size_t accepted = 0;
  auto input = static_cast<std::uint32_t>(starting_token);
  double begin = monotonic_milliseconds();
  auto target = session.value().greedy_step(input);
  target_ms += monotonic_milliseconds() - begin;
  if (!target.ok()) return fail(target.status());
  begin = monotonic_milliseconds();
  auto candidate = session.value().mtp_draft(target.value());
  draft_ms += monotonic_milliseconds() - begin;
  if (!candidate.ok()) return fail(candidate.status());
  std::vector<std::uint32_t> target_tokens;
  std::vector<std::uint32_t> draft_tokens;
  target_tokens.reserve(steps);
  draft_tokens.reserve(steps);
  for (std::size_t step = 0; step < steps; ++step) {
    begin = monotonic_milliseconds();
    auto next_target = session.value().greedy_step(target.value());
    target_ms += monotonic_milliseconds() - begin;
    if (!next_target.ok()) return fail(next_target.status());
    target_tokens.push_back(next_target.value());
    draft_tokens.push_back(candidate.value());
    if (next_target.value() == candidate.value()) ++accepted;
    target = std::move(next_target);
    begin = monotonic_milliseconds();
    candidate = session.value().mtp_draft(target.value());
    draft_ms += monotonic_milliseconds() - begin;
    if (!candidate.ok()) return fail(candidate.status());
  }
  auto target_text = tokenizer.value().decode(target_tokens);
  if (!target_text.ok()) return fail(target_text.status());
  std::cout << std::fixed << std::setprecision(3)
            << "built-in MTP acceptance oracle passed\n"
            << "  comparisons:     " << steps << '\n'
            << "  exact matches:   " << accepted << '\n'
            << "  acceptance:      " << accepted * 100.0 / steps << "%\n"
            << "  target speed:    " << (steps + 1) * 1000.0 / target_ms << " steps/s\n"
            << "  MTP draft speed: " << (steps + 1) * 1000.0 / draft_ms << " steps/s\n"
            << "  target bytes:    " << std::quoted(target_text.value()) << '\n'
            << "  first pairs:     ";
  const auto pairs = std::min<std::size_t>(steps, 12);
  for (std::size_t index = 0; index < pairs; ++index) {
    if (index != 0) std::cout << ' ';
    std::cout << draft_tokens[index] << (draft_tokens[index] == target_tokens[index] ? "=" : "!=")
              << target_tokens[index];
  }
  std::cout << '\n';
  return 0;
}

struct BenchOptions {
  Quantization format{Quantization::w2g128};
  std::size_t rows{17408};
  std::size_t columns{5120};
  int iterations{100};
  int device{0};
  unsigned batch{1};
  unsigned activation_bits{0};
};

bool parse_positive(std::string_view value, std::size_t& output) {
  try {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(std::string(value), &consumed);
    if (consumed != value.size() || parsed == 0) return false;
    output = static_cast<std::size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_nonnegative(std::string_view value, std::size_t& output) {
  try {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(std::string(value), &consumed);
    if (consumed != value.size()) return false;
    output = static_cast<std::size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_activation_mode(
    std::string_view value,
    gfxinfer::ActivationMode& output) {
  if (value == "f16") output = gfxinfer::ActivationMode::fp16;
  else if (value == "a8") output = gfxinfer::ActivationMode::int8;
  else if (value == "a4") output = gfxinfer::ActivationMode::int4;
  else if (value == "a4w2") output = gfxinfer::ActivationMode::int4_w2;
  else if (value == "a4w4") output = gfxinfer::ActivationMode::int4_w4;
  else return false;
  return true;
}

gfxinfer::Result<BenchOptions> parse_bench_options(int argc, char** argv) {
  BenchOptions options;
  for (int index = 2; index < argc; index += 2) {
    if (index + 1 >= argc) {
      return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument,
                              std::string("missing value for ") + argv[index]};
    }
    const std::string_view option = argv[index];
    const std::string_view value = argv[index + 1];
    if (option == "--format") {
      if (value == "w2") options.format = Quantization::w2g128;
      else if (value == "w3") options.format = Quantization::w3g128;
      else if (value == "w4") options.format = Quantization::w4g128;
      else return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument,
                                   "format must be w2, w3, or w4"};
    } else if (option == "--rows") {
      if (!parse_positive(value, options.rows))
        return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument, "invalid row count"};
    } else if (option == "--cols") {
      if (!parse_positive(value, options.columns))
        return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument, "invalid column count"};
    } else if (option == "--iters") {
      std::size_t parsed = 0;
      if (!parse_positive(value, parsed) || parsed > 100000)
        return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument, "invalid iteration count"};
      options.iterations = static_cast<int>(parsed);
    } else if (option == "--device") {
      std::size_t parsed = 0;
      if (!parse_positive(value, parsed) && value != "0")
        return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument, "invalid device index"};
      options.device = static_cast<int>(parsed);
    } else if (option == "--batch") {
      std::size_t parsed = 0;
      if (!parse_positive(value, parsed) || parsed > 32)
        return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument,
                                "batch must be in [1, 8]"};
      options.batch = static_cast<unsigned>(parsed);
    } else if (option == "--kernel") {
      if (value == "scalar") options.activation_bits = 0;
      else if (value == "wmma-a4") options.activation_bits = 4;
      else if (value == "wmma-a8") options.activation_bits = 8;
      else if (value == "wmma-f16") options.activation_bits = 16;
      else return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument,
                                   "kernel must be scalar, wmma-a4, wmma-a8, or wmma-f16"};
    } else {
      return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument,
                              "unknown benchmark option: " + std::string(option)};
    }
  }
  if (options.columns % 128 != 0) {
    return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument,
                            "benchmark columns must be divisible by 128"};
  }
  if (options.activation_bits != 0 &&
      (options.rows % 16 != 0 ||
       (options.format == Quantization::w3g128 && options.activation_bits != 4))) {
    return gfxinfer::Status{gfxinfer::ErrorCode::invalid_argument,
                            "WMMA requires N divisible by 16; W3 supports only wmma-a4"};
  }
  return options;
}

int bench_wmma(const BenchOptions& options) {
  auto device = gfxinfer::require_gfx1200_device(options.device);
  if (!device.ok()) return fail(device.status());

  const auto row_bytes = gfxinfer::packed_row_bytes(options.format, options.columns);
  const auto groups = options.columns / 128;
  std::mt19937 generator(0x1200U);
  std::uniform_int_distribution<int> byte_distribution(0, 255);
  std::uniform_real_distribution<float> scale_distribution(0.002F, 0.08F);
  std::uniform_real_distribution<float> input_distribution(-2.0F, 2.0F);

  gfxinfer::PackedMatrix host_matrix;
  host_matrix.rows = options.rows;
  host_matrix.columns = options.columns;
  host_matrix.format = options.format;
  host_matrix.codes.resize(options.rows * row_bytes);
  host_matrix.scales.resize(options.rows * groups);
  for (auto& code : host_matrix.codes) {
    code = static_cast<std::uint8_t>(byte_distribution(generator));
  }
  for (auto& scale : host_matrix.scales) {
    scale = __half2float(__float2half(scale_distribution(generator)));
  }
  auto tiled = gfxinfer::tile_group128_for_gfx12(host_matrix);
  if (!tiled.ok()) return fail(tiled.status());
  std::vector<__half> tiled_scales(tiled.value().scales.size());
  for (std::size_t index = 0; index < tiled_scales.size(); ++index) {
    tiled_scales[index] = __float2half(tiled.value().scales[index]);
  }
  std::vector<__half> host_input(options.columns * options.batch);
  for (auto& value : host_input) value = __float2half(input_distribution(generator));

  DeviceBuffer<std::uint8_t> device_weights;
  DeviceBuffer<__half> device_weight_scales;
  DeviceBuffer<__half> device_input;
  DeviceBuffer<std::uint8_t> device_activation_codes;
  DeviceBuffer<__half> device_activation_scales;
  DeviceBuffer<float> device_output;
  for (auto status : {
           device_weights.allocate(tiled.value().codes.size()),
           device_weight_scales.allocate(tiled_scales.size()),
           device_input.allocate(host_input.size()),
           device_activation_codes.allocate(
               options.batch * options.columns * options.activation_bits / 8),
           device_activation_scales.allocate(options.batch * groups),
           device_output.allocate(options.rows * options.batch)}) {
    if (!status.ok()) return fail(status);
  }
  for (auto status : {
           gfxinfer::hip_status(
               hipMemcpy(device_weights.get(), tiled.value().codes.data(), device_weights.bytes(),
                         hipMemcpyHostToDevice),
               "tiled weight upload failed"),
           gfxinfer::hip_status(
               hipMemcpy(device_weight_scales.get(), tiled_scales.data(),
                         device_weight_scales.bytes(), hipMemcpyHostToDevice),
               "tiled scale upload failed"),
           gfxinfer::hip_status(
               hipMemcpy(device_input.get(), host_input.data(), device_input.bytes(),
                         hipMemcpyHostToDevice),
               "input upload failed")}) {
    if (!status.ok()) return fail(status);
  }

  const auto launch_once = [&]() -> gfxinfer::Status {
    if (options.activation_bits == 16) {
      return gfxinfer::launch_gfx12_f16_gemm(
          options.format, device_weights.get(), device_weight_scales.get(),
          device_input.get(), device_output.get(), options.rows,
          options.columns, options.batch, nullptr);
    }
    if (options.activation_bits == 4) {
      auto status = gfxinfer::launch_quantize_activation_i4(
          device_input.get(), device_activation_codes.get(), device_activation_scales.get(),
          options.columns, options.batch, nullptr);
      if (!status.ok()) return status;
      return gfxinfer::launch_gfx12_i4_gemm(
          options.format, device_weights.get(), device_weight_scales.get(),
          device_activation_codes.get(), device_activation_scales.get(), device_output.get(),
          options.rows, options.columns, options.batch, true, nullptr);
    }
    auto status = gfxinfer::launch_quantize_activation_i8(
        device_input.get(), reinterpret_cast<std::int8_t*>(device_activation_codes.get()),
        device_activation_scales.get(), options.columns, options.batch, nullptr);
    if (!status.ok()) return status;
    return gfxinfer::launch_gfx12_i8_gemm(
        options.format, device_weights.get(), device_weight_scales.get(),
        reinterpret_cast<const std::int8_t*>(device_activation_codes.get()),
        device_activation_scales.get(), device_output.get(), options.rows,
        options.columns, options.batch, nullptr);
  };
  for (int iteration = 0; iteration < 10; ++iteration) {
    if (auto status = launch_once(); !status.ok()) return fail(status);
  }
  if (auto status = gfxinfer::hip_status(hipDeviceSynchronize(), "WMMA warmup failed");
      !status.ok()) return fail(status);

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (auto status = gfxinfer::hip_status(hipEventCreate(&start), "event creation failed");
      !status.ok()) return fail(status);
  if (auto status = gfxinfer::hip_status(hipEventCreate(&stop), "event creation failed");
      !status.ok()) {
    (void)hipEventDestroy(start);
    return fail(status);
  }
  (void)hipEventRecord(start);
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    if (auto status = launch_once(); !status.ok()) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(stop);
      return fail(status);
    }
  }
  (void)hipEventRecord(stop);
  if (auto status = gfxinfer::hip_status(hipEventSynchronize(stop), "event wait failed");
      !status.ok()) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(status);
  }
  float elapsed_ms = 0.0F;
  if (auto status = gfxinfer::hip_status(
          hipEventElapsedTime(&elapsed_ms, start, stop), "event timing failed"); !status.ok()) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(status);
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);

  std::vector<float> output(options.rows * options.batch);
  std::vector<std::uint8_t> activation_codes(
      options.batch * options.columns * options.activation_bits / 8);
  std::vector<__half> activation_scales(options.batch * groups);
  for (auto status : {
           gfxinfer::hip_status(
               hipMemcpy(output.data(), device_output.get(), device_output.bytes(),
                         hipMemcpyDeviceToHost),
               "output download failed"),
           gfxinfer::hip_status(
               hipMemcpy(activation_codes.data(), device_activation_codes.get(),
                         activation_codes.size(), hipMemcpyDeviceToHost),
               "quantized activation download failed"),
           gfxinfer::hip_status(
               hipMemcpy(activation_scales.data(), device_activation_scales.get(),
                         activation_scales.size() * sizeof(__half), hipMemcpyDeviceToHost),
               "activation scale download failed")}) {
    if (!status.ok()) return fail(status);
  }

  const std::array<std::size_t, 3> check_rows{0, options.rows / 2, options.rows - 1};
  float maximum_error = 0.0F;
  for (unsigned batch = 0; batch < options.batch; ++batch) {
    for (const auto row : check_rows) {
      float expected = 0.0F;
      for (std::size_t column = 0; column < options.columns; ++column) {
        int quantized = 0;
        if (options.activation_bits == 4) {
          const auto packed = activation_codes[static_cast<std::size_t>(batch) *
                                                   (options.columns / 2) +
                                               column / 2];
          const auto code = static_cast<int>((packed >> ((column & 1U) * 4U)) & 0x0fU);
          quantized = code >= 8 ? code - 16 : code;
        } else if (options.activation_bits == 8) {
          quantized = static_cast<std::int8_t>(
              activation_codes[static_cast<std::size_t>(batch) * options.columns + column]);
        }
        const float activation = options.activation_bits == 16
                                     ? __half2float(host_input[
                                           static_cast<std::size_t>(batch) * options.columns +
                                           column])
                                     : static_cast<float>(quantized) *
                                           __half2float(activation_scales[
                                               static_cast<std::size_t>(batch) * groups +
                                               column / 128]);
        expected = std::fma(gfxinfer::dequantized_weight(host_matrix, row, column),
                            activation, expected);
      }
      const float actual = output[static_cast<std::size_t>(batch) * options.rows + row];
      const float error = std::abs(expected - actual);
      maximum_error = std::max(maximum_error, error);
      const float tolerance = 0.008F * std::max(1.0F, std::abs(expected));
      if (!std::isfinite(actual) || error > tolerance) {
        std::cerr << "error: WMMA validation failed at batch " << batch << " row " << row
                  << ": expected=" << expected << " actual=" << actual
                  << " tolerance=" << tolerance << '\n';
        return 1;
      }
    }
  }

  const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
  const double launches_per_second = static_cast<double>(options.iterations) / seconds;
  const double packed_gbps = static_cast<double>(host_matrix.codes.size()) *
                             static_cast<double>(options.iterations) / seconds / 1.0e9;
  std::cout << std::fixed << std::setprecision(3)
            << "device:            " << device.value().name << " (" << device.value().architecture
            << ")\nformat:            " << gfxinfer::quantization_name(options.format)
            << "\nkernel:            gfx12-wmma-"
            << (options.activation_bits == 16 ? "f16" : "a" + std::to_string(options.activation_bits))
            << "\nshape:             [" << options.rows << ',' << options.columns << "]"
            << "\nbatch:             " << options.batch
            << "\npacked weights:    " << host_matrix.codes.size() << " bytes"
            << "\nmean quant+GEMM:   " << elapsed_ms / options.iterations << " ms"
            << "\nlaunches/second:   " << launches_per_second
            << "\npacked bandwidth:  " << packed_gbps << " GB/s"
            << "\neffective vectors: " << launches_per_second * options.batch << " /s"
            << "\nmax sampled error: " << maximum_error << "\nvalidation:        pass\n";
  return 0;
}

int bench_gemv(const BenchOptions& options) {
  if (options.activation_bits != 0) return bench_wmma(options);
  auto device = gfxinfer::require_gfx1200_device(options.device);
  if (!device.ok()) return fail(device.status());

  const auto row_bytes = gfxinfer::packed_row_bytes(options.format, options.columns);
  const auto groups = options.columns / 128;
  if (options.rows > std::numeric_limits<std::size_t>::max() / row_bytes) {
    std::cerr << "error: benchmark allocation size overflow\n";
    return 1;
  }
  std::mt19937 generator(0x1200U);
  std::uniform_int_distribution<int> byte_distribution(0, 255);
  std::uniform_real_distribution<float> scale_distribution(0.002F, 0.08F);
  std::uniform_real_distribution<float> input_distribution(-2.0F, 2.0F);

  gfxinfer::PackedMatrix host_matrix;
  host_matrix.rows = options.rows;
  host_matrix.columns = options.columns;
  host_matrix.format = options.format;
  host_matrix.codes.resize(options.rows * row_bytes);
  host_matrix.scales.resize(options.rows * groups);
  for (auto& code : host_matrix.codes) code = static_cast<std::uint8_t>(byte_distribution(generator));

  std::vector<__half> host_scales(host_matrix.scales.size());
  for (std::size_t index = 0; index < host_scales.size(); ++index) {
    host_scales[index] = __float2half(scale_distribution(generator));
    host_matrix.scales[index] = __half2float(host_scales[index]);
  }
  std::vector<__half> host_input(options.columns * options.batch);
  std::vector<float> reference_input(options.columns * options.batch);
  for (std::size_t index = 0; index < host_input.size(); ++index) {
    host_input[index] = __float2half(input_distribution(generator));
    reference_input[index] = __half2float(host_input[index]);
  }

  DeviceBuffer<std::uint8_t> device_weights;
  DeviceBuffer<__half> device_scales;
  DeviceBuffer<__half> device_input;
  DeviceBuffer<float> device_output;
  for (auto status : {device_weights.allocate(host_matrix.codes.size()),
                      device_scales.allocate(host_scales.size()),
                      device_input.allocate(host_input.size()),
                      device_output.allocate(options.rows * options.batch)}) {
    if (!status.ok()) return fail(status);
  }
  for (auto status : {
           gfxinfer::hip_status(hipMemcpy(device_weights.get(), host_matrix.codes.data(),
                                          device_weights.bytes(), hipMemcpyHostToDevice),
                                "weight upload failed"),
           gfxinfer::hip_status(hipMemcpy(device_scales.get(), host_scales.data(),
                                          device_scales.bytes(), hipMemcpyHostToDevice),
                                "scale upload failed"),
           gfxinfer::hip_status(hipMemcpy(device_input.get(), host_input.data(),
                                          device_input.bytes(), hipMemcpyHostToDevice),
                                "input upload failed")}) {
    if (!status.ok()) return fail(status);
  }

  for (int iteration = 0; iteration < 10; ++iteration) {
    const auto status = options.batch == 1
                            ? gfxinfer::launch_lowbit_gemv(
                                  options.format, device_weights.get(), device_scales.get(),
                                  device_input.get(), device_output.get(), options.rows,
                                  options.columns, nullptr)
                            : gfxinfer::launch_lowbit_gemv_batch(
                                  options.format, device_weights.get(), device_scales.get(),
                                  device_input.get(), device_output.get(), options.rows,
                                  options.columns, options.batch, nullptr);
    if (!status.ok()) return fail(status);
  }
  if (auto status = gfxinfer::hip_status(hipDeviceSynchronize(), "warmup failed"); !status.ok())
    return fail(status);

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (auto status = gfxinfer::hip_status(hipEventCreate(&start), "event creation failed");
      !status.ok()) return fail(status);
  if (auto status = gfxinfer::hip_status(hipEventCreate(&stop), "event creation failed");
      !status.ok()) {
    (void)hipEventDestroy(start);
    return fail(status);
  }
  if (auto status = gfxinfer::hip_status(hipEventRecord(start), "start event record failed");
      !status.ok()) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(status);
  }
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    const auto status = options.batch == 1
                            ? gfxinfer::launch_lowbit_gemv(
                                  options.format, device_weights.get(), device_scales.get(),
                                  device_input.get(), device_output.get(), options.rows,
                                  options.columns, nullptr)
                            : gfxinfer::launch_lowbit_gemv_batch(
                                  options.format, device_weights.get(), device_scales.get(),
                                  device_input.get(), device_output.get(), options.rows,
                                  options.columns, options.batch, nullptr);
    if (!status.ok()) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(stop);
      return fail(status);
    }
  }
  if (auto status = gfxinfer::hip_status(hipEventRecord(stop), "stop event record failed");
      !status.ok()) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(status);
  }
  if (auto status = gfxinfer::hip_status(hipEventSynchronize(stop), "event wait failed");
      !status.ok()) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(status);
  }
  float elapsed_ms = 0.0F;
  if (auto status = gfxinfer::hip_status(
          hipEventElapsedTime(&elapsed_ms, start, stop), "event timing failed"); !status.ok()) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return fail(status);
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);

  std::vector<float> output(options.rows * options.batch);
  if (auto status = gfxinfer::hip_status(
          hipMemcpy(output.data(), device_output.get(), device_output.bytes(), hipMemcpyDeviceToHost),
          "output download failed"); !status.ok()) return fail(status);

  const std::array<std::size_t, 3> check_rows{0, options.rows / 2, options.rows - 1};
  float maximum_error = 0.0F;
  for (unsigned batch = 0; batch < options.batch; ++batch) {
    for (const auto row : check_rows) {
      float expected = 0.0F;
      for (std::size_t column = 0; column < options.columns; ++column) {
        expected = std::fma(
            gfxinfer::dequantized_weight(host_matrix, row, column),
            reference_input[static_cast<std::size_t>(batch) * options.columns + column],
            expected);
      }
      const auto actual = output[static_cast<std::size_t>(batch) * options.rows + row];
      maximum_error = std::max(maximum_error, std::abs(expected - actual));
      const float tolerance = 0.003F * std::max(1.0F, std::abs(expected));
      if (!std::isfinite(actual) || std::abs(expected - actual) > tolerance) {
        std::cerr << "error: validation failed at batch " << batch << " row " << row
                  << ": expected=" << expected << " actual=" << actual
                  << " tolerance=" << tolerance << '\n';
        return 1;
      }
    }
  }

  const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
  const double launches_per_second = static_cast<double>(options.iterations) / seconds;
  const double packed_gbps = static_cast<double>(host_matrix.codes.size()) *
                             static_cast<double>(options.iterations) / seconds / 1.0e9;
  std::cout << std::fixed << std::setprecision(3)
            << "device:            " << device.value().name << " (" << device.value().architecture
            << ")\nformat:            " << gfxinfer::quantization_name(options.format)
            << "\nshape:             [" << options.rows << ',' << options.columns << "]"
            << "\nbatch:             " << options.batch
            << "\npacked weights:    " << host_matrix.codes.size() << " bytes"
            << "\nmean launch:       " << elapsed_ms / options.iterations << " ms"
            << "\nlaunches/second:   " << launches_per_second
            << "\npacked bandwidth:  " << packed_gbps << " GB/s"
            << "\neffective vectors: " << launches_per_second * options.batch << " /s"
            << "\nmax sampled error: " << maximum_error << "\nvalidation:        pass\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 1;
  }
  const std::string_view command = argv[1];
  if (command == "probe") return probe();
  if (command == "inspect") {
    if (argc != 3) {
      print_usage();
      return 1;
    }
    return inspect(argv[2]);
  }
  if (command == "verify-source") {
    if (argc != 3) {
      print_usage();
      return 1;
    }
    return verify_source(argv[2]);
  }
  if (command == "inspect-safetensors") {
    if (argc != 3) {
      print_usage();
      return 1;
    }
    return inspect_safetensors(argv[2]);
  }
  if (command == "convert") return convert(argc, argv);
  if (command == "load-check") {
    if (argc != 3) {
      print_usage();
      return 1;
    }
    return load_check(argv[2]);
  }
  if (command == "model-smoke") {
    if (argc < 3 || argc > 7 || ((argc - 3) % 2) != 0) {
      print_usage();
      return 1;
    }
    std::size_t token = 248044;
    std::size_t iterations = 50;
    for (int index = 3; index < argc; index += 2) {
      const std::string_view option = argv[index];
      if (index + 1 >= argc || !parse_positive(argv[index + 1],
                                                option == "--token" ? token : iterations) ||
          (option != "--token" && option != "--iters")) {
        std::cerr << "error: invalid model-smoke option\n";
        return 1;
      }
    }
    if (iterations > 10000) {
      std::cerr << "error: model-smoke iterations exceed 10000\n";
      return 1;
    }
    return model_smoke(argv[2], token, static_cast<int>(iterations));
  }
  if (command == "gdn-smoke") {
    if (argc < 3 || argc > 7 || ((argc - 3) % 2) != 0) {
      print_usage();
      return 1;
    }
    std::size_t token = 248044;
    std::size_t iterations = 50;
    for (int index = 3; index < argc; index += 2) {
      const std::string_view option = argv[index];
      if (index + 1 >= argc || !parse_positive(argv[index + 1],
                                                option == "--token" ? token : iterations) ||
          (option != "--token" && option != "--iters")) {
        std::cerr << "error: invalid gdn-smoke option\n";
        return 1;
      }
    }
    if (iterations > 10000) {
      std::cerr << "error: gdn-smoke iterations exceed 10000\n";
      return 1;
    }
    return gdn_smoke(argv[2], token, static_cast<int>(iterations));
  }
  if (command == "attention-smoke") {
    if (argc < 3 || argc > 7 || ((argc - 3) % 2) != 0) {
      print_usage();
      return 1;
    }
    std::size_t token = 248044;
    std::size_t iterations = 50;
    for (int index = 3; index < argc; index += 2) {
      const std::string_view option = argv[index];
      if (index + 1 >= argc || !parse_positive(argv[index + 1],
                                                option == "--token" ? token : iterations) ||
          (option != "--token" && option != "--iters")) {
        std::cerr << "error: invalid attention-smoke option\n";
        return 1;
      }
    }
    if (iterations > 4096) {
      std::cerr << "error: attention-smoke iterations exceed 4096\n";
      return 1;
    }
    return attention_smoke(argv[2], token, static_cast<int>(iterations));
  }
  if (command == "decode-bench") {
    if (argc < 3 || argc > 9 || ((argc - 3) % 2) != 0) {
      print_usage();
      return 1;
    }
    std::size_t token = 248044;
    std::size_t tokens = 16;
    std::size_t context = 512;
    for (int index = 3; index < argc; index += 2) {
      const std::string_view option = argv[index];
      std::size_t* destination = nullptr;
      if (option == "--token") destination = &token;
      else if (option == "--tokens") destination = &tokens;
      else if (option == "--context") destination = &context;
      if (destination == nullptr || index + 1 >= argc ||
          !parse_positive(argv[index + 1], *destination)) {
        std::cerr << "error: invalid decode-bench option\n";
        return 1;
      }
    }
    if (token >= gfxinfer::Qwen38Spec::vocabulary_size || tokens > context) {
      std::cerr << "error: decode-bench token or context range is invalid\n";
      return 1;
    }
    return decode_bench(argv[2], token, tokens, context);
  }
  if (command == "block-bench") {
    if (argc < 3 || argc > 13 || ((argc - 3) % 2) != 0) {
      print_usage();
      return 1;
    }
    std::size_t token = 248044;
    std::size_t batch = 8;
    std::size_t blocks = 4;
    std::size_t context = 128;
    gfxinfer::ActivationMode activation_mode = gfxinfer::ActivationMode::fp16;
    for (int index = 3; index < argc; index += 2) {
      const std::string_view option = argv[index];
      if (option == "--activation") {
        if (index + 1 >= argc ||
            !parse_activation_mode(argv[index + 1], activation_mode)) {
          std::cerr << "error: invalid block-bench activation mode\n";
          return 1;
        }
        continue;
      }
      std::size_t* destination = nullptr;
      if (option == "--token") destination = &token;
      else if (option == "--batch") destination = &batch;
      else if (option == "--blocks") destination = &blocks;
      else if (option == "--context") destination = &context;
      if (destination == nullptr || index + 1 >= argc ||
          !parse_positive(argv[index + 1], *destination)) {
        std::cerr << "error: invalid block-bench option\n";
        return 1;
      }
    }
    if (token >= gfxinfer::Qwen38Spec::tokenizer_vocabulary_size || batch > 32 ||
        (blocks + 1) * batch > context ||
        context > gfxinfer::Qwen38Spec::maximum_context_tokens) {
      std::cerr << "error: block-bench token, batch, or context range is invalid\n";
      return 1;
    }
    return block_bench(argv[2], token, batch, blocks, context, activation_mode);
  }
  if (command == "spec-bench") {
    if (argc < 3 || argc > 13 || ((argc - 3) % 2) != 0) {
      print_usage();
      return 1;
    }
    std::size_t token = 248044;
    std::size_t drafts = 3;
    std::size_t tokens = 32;
    std::size_t context = 128;
    gfxinfer::ActivationMode activation_mode = gfxinfer::ActivationMode::fp16;
    for (int index = 3; index < argc; index += 2) {
      const std::string_view option = argv[index];
      if (option == "--activation") {
        if (index + 1 >= argc ||
            !parse_activation_mode(argv[index + 1], activation_mode)) {
          std::cerr << "error: invalid spec-bench activation mode\n";
          return 1;
        }
        continue;
      }
      std::size_t* destination = nullptr;
      if (option == "--token") destination = &token;
      else if (option == "--drafts") destination = &drafts;
      else if (option == "--tokens") destination = &tokens;
      else if (option == "--context") destination = &context;
      if (destination == nullptr || index + 1 >= argc ||
          !parse_positive(argv[index + 1], *destination)) {
        std::cerr << "error: invalid spec-bench option\n";
        return 1;
      }
    }
    if (token >= gfxinfer::Qwen38Spec::tokenizer_vocabulary_size ||
        drafts > 31 || tokens + drafts + 1 > context ||
        context > gfxinfer::Qwen38Spec::maximum_context_tokens) {
      std::cerr << "error: spec-bench token, draft, or context range is invalid\n";
      return 1;
    }
    return speculative_bench(
        argv[2], token, drafts, tokens, context, activation_mode);
  }
  if (command == "tokenizer-check") {
    if (argc != 4) {
      print_usage();
      return 1;
    }
    return tokenizer_check(argv[2], argv[3]);
  }
  if (command == "generate") {
    if (argc < 5 || argc > 13 || ((argc - 3) % 2) != 0) {
      print_usage();
      return 1;
    }
    std::string_view prompt;
    std::size_t maximum_tokens = 32;
    std::size_t context = 4096;
    std::size_t draft_count = 4;
    gfxinfer::ActivationMode activation_mode = gfxinfer::ActivationMode::int4;
    for (int index = 3; index < argc; index += 2) {
      const std::string_view option = argv[index];
      if (option == "--prompt") {
        prompt = argv[index + 1];
      } else if (option == "--max-tokens") {
        if (!parse_positive(argv[index + 1], maximum_tokens)) {
          std::cerr << "error: invalid maximum token count\n";
          return 1;
        }
      } else if (option == "--context") {
        if (!parse_positive(argv[index + 1], context)) {
          std::cerr << "error: invalid context length\n";
          return 1;
        }
      } else if (option == "--activation") {
        if (!parse_activation_mode(argv[index + 1], activation_mode)) {
          std::cerr << "error: invalid generate activation mode\n";
          return 1;
        }
      } else if (option == "--drafts") {
        if (!parse_nonnegative(argv[index + 1], draft_count) || draft_count > 31) {
          std::cerr << "error: invalid speculative draft count\n";
          return 1;
        }
      } else {
        std::cerr << "error: invalid generate option\n";
        return 1;
      }
    }
    if (prompt.empty() || maximum_tokens > context ||
        context > gfxinfer::Qwen38Spec::maximum_context_tokens) {
      std::cerr << "error: prompt or generation range is invalid\n";
      return 1;
    }
    return generate(
        argv[2], prompt, maximum_tokens, context, activation_mode, draft_count);
  }
  if (command == "mtp-bench") {
    if (argc < 3 || argc > 11 || ((argc - 3) % 2) != 0) {
      print_usage();
      return 1;
    }
    std::size_t token = 248044;
    std::size_t steps = 32;
    std::size_t context = 512;
    gfxinfer::ActivationMode activation_mode = gfxinfer::ActivationMode::fp16;
    for (int index = 3; index < argc; index += 2) {
      const std::string_view option = argv[index];
      if (option == "--activation") {
        if (index + 1 >= argc ||
            !parse_activation_mode(argv[index + 1], activation_mode)) {
          std::cerr << "error: invalid mtp-bench activation mode\n";
          return 1;
        }
        continue;
      }
      std::size_t* destination = nullptr;
      if (option == "--token") destination = &token;
      else if (option == "--steps") destination = &steps;
      else if (option == "--context") destination = &context;
      if (destination == nullptr || index + 1 >= argc ||
          !parse_positive(argv[index + 1], *destination)) {
        std::cerr << "error: invalid mtp-bench option\n";
        return 1;
      }
    }
    if (token >= gfxinfer::Qwen38Spec::tokenizer_vocabulary_size || steps + 1 > context) {
      std::cerr << "error: MTP benchmark token or context range is invalid\n";
      return 1;
    }
    return mtp_bench(argv[2], token, steps, context, activation_mode);
  }
  if (command == "bench-gemv") {
    auto options = parse_bench_options(argc, argv);
    if (!options.ok()) return fail(options.status());
    return bench_gemv(options.value());
  }
  if (command == "--version" || command == "version") {
    std::cout << gfxinfer::kVersion << '\n';
    return 0;
  }
  print_usage();
  return 1;
}
