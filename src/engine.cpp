// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/engine.h"

#include <algorithm>
#include <limits>
#include <span>
#include <utility>

#include <hip/hip_runtime_api.h>

#include "gfxinfer/version.h"

namespace gfxinfer {

struct Engine::Impl {
  ~Impl() {
    if (config.device_index >= 0) (void)hipSetDevice(config.device_index);
    if (stream != nullptr) (void)hipStreamDestroy(stream);
    if (device_arena != nullptr) (void)hipFree(device_arena);
  }

  EngineConfig config;
  ArtifactView artifact;
  LoadSummary summary;
  std::byte* device_arena{nullptr};
  hipStream_t stream{nullptr};
  std::vector<DeviceTensorView> device_tensors;
};

namespace {

[[nodiscard]] bool is_resource(std::string_view name) {
  return name.starts_with("resources/");
}

[[nodiscard]] Result<std::size_t> aligned_size(
    std::size_t cursor,
    std::uint64_t bytes,
    std::uint32_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1U)) != 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status{ErrorCode::invalid_argument, "invalid resident tensor alignment or size"};
  }
  const auto mask = static_cast<std::size_t>(alignment - 1U);
  if (cursor > std::numeric_limits<std::size_t>::max() - mask) {
    return Status{ErrorCode::out_of_memory, "resident arena offset overflow"};
  }
  cursor = (cursor + mask) & ~mask;
  if (static_cast<std::size_t>(bytes) > std::numeric_limits<std::size_t>::max() - cursor) {
    return Status{ErrorCode::out_of_memory, "resident arena size overflow"};
  }
  return cursor + static_cast<std::size_t>(bytes);
}

}  // namespace

Engine::Engine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

Result<Engine> Engine::create(EngineConfig config) {
  if (config.artifact.empty()) {
    return Status{ErrorCode::invalid_argument, "artifact path is required"};
  }
  if (config.maximum_concurrency == 0 || config.maximum_concurrency > 8) {
    return Status{ErrorCode::invalid_argument, "maximum concurrency must be in [1, 8]"};
  }
  if (config.maximum_context_tokens == 0 || config.maximum_context_tokens > 262144) {
    return Status{ErrorCode::invalid_argument,
                  "maximum context must be in [1, 262144] tokens"};
  }
  if (config.maximum_mtp_depth > 8) {
    return Status{ErrorCode::invalid_argument, "maximum MTP depth must be in [0, 8]"};
  }

  auto device = require_gfx1200_device(config.device_index);
  if (!device.ok()) {
    return device.status();
  }
  auto artifact = ArtifactView::open(
      config.artifact, {.verify_checksums = config.verify_artifact_checksums});
  if (!artifact.ok()) {
    return artifact.status();
  }
  const auto& identity = artifact.value().identity();
  if (identity.target_arch != kRequiredArchitecture) {
    return Status{ErrorCode::unsupported,
                  "artifact target is " + identity.target_arch + "; expected gfx1200"};
  }
  if (identity.model_id != kModelId) {
    return Status{ErrorCode::unsupported,
                  "artifact model is " + identity.model_id + "; expected qwen3.8-27b"};
  }

  constexpr std::size_t reserve_bytes = 512ULL * 1024ULL * 1024ULL;
  const auto maximum_budget = device.value().total_memory_bytes > reserve_bytes
                                  ? device.value().total_memory_bytes - reserve_bytes
                                  : 0;
  if (config.memory_budget_bytes == 0) {
    config.memory_budget_bytes = maximum_budget;
  }
  if (config.memory_budget_bytes > maximum_budget) {
    return Status{ErrorCode::invalid_argument,
                  "memory budget must leave at least 512 MiB outside the engine"};
  }

  auto impl = std::make_unique<Impl>();
  impl->config = std::move(config);
  std::size_t device_free_before = 0;
  std::size_t device_total = 0;
  if (const auto error = hipMemGetInfo(&device_free_before, &device_total);
      error != hipSuccess) {
    return hip_status(error, "hipMemGetInfo before weight load failed");
  }
  impl->summary = {
      identity,
      device.value(),
      artifact.value().file_bytes(),
      impl->config.memory_budget_bytes,
      artifact.value().tensors().size(),
      0,
      0,
      device_free_before,
      device_free_before,
  };
  impl->artifact = std::move(artifact).value();

  std::size_t arena_bytes = 0;
  struct Placement {
    const TensorDescriptor* tensor;
    std::size_t data_offset;
    std::size_t auxiliary_offset;
  };
  std::vector<Placement> placements;
  placements.reserve(impl->artifact.tensors().size());
  for (const auto& tensor : impl->artifact.tensors()) {
    if (is_resource(tensor.name)) continue;
    const auto data_begin = aligned_size(arena_bytes, 0, tensor.alignment);
    if (!data_begin.ok()) return data_begin.status();
    const auto data_end = aligned_size(data_begin.value(), tensor.data_bytes, 1);
    if (!data_end.ok()) return data_end.status();
    std::size_t auxiliary_begin = 0;
    arena_bytes = data_end.value();
    if (tensor.aux_bytes != 0) {
      const auto begin = aligned_size(arena_bytes, 0, tensor.alignment);
      if (!begin.ok()) return begin.status();
      auxiliary_begin = begin.value();
      const auto end = aligned_size(auxiliary_begin, tensor.aux_bytes, 1);
      if (!end.ok()) return end.status();
      arena_bytes = end.value();
    }
    placements.push_back({&tensor, data_begin.value(), auxiliary_begin});
  }
  impl->summary.resident_tensor_count = placements.size();
  impl->summary.resident_weight_bytes = arena_bytes;
  if (arena_bytes > impl->config.memory_budget_bytes) {
    return Status{ErrorCode::out_of_memory,
                  "resident weights exceed the configured GPU memory budget"};
  }
  if (!impl->config.upload_weights) return Engine(std::move(impl));
  if (arena_bytes > device_free_before) {
    return Status{ErrorCode::out_of_memory,
                  "resident weights exceed currently free GPU memory"};
  }
  if (const auto error = hipMalloc(reinterpret_cast<void**>(&impl->device_arena), arena_bytes);
      error != hipSuccess) {
    return hip_status(error, "resident weight arena allocation failed");
  }
  if (const auto error = hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking);
      error != hipSuccess) {
    return hip_status(error, "engine compute stream creation failed");
  }
  impl->device_tensors.reserve(placements.size());
  for (const auto& placement : placements) {
    const auto& tensor = *placement.tensor;
    auto* device_data = impl->device_arena + placement.data_offset;
    if (const auto error = hipMemcpyAsync(
            device_data, impl->artifact.data(tensor).data(), tensor.data_bytes,
            hipMemcpyHostToDevice, impl->stream);
        error != hipSuccess) {
      return hip_status(error, "weight upload failed for " + tensor.name);
    }
    std::byte* device_auxiliary = nullptr;
    if (tensor.aux_bytes != 0) {
      device_auxiliary = impl->device_arena + placement.auxiliary_offset;
      if (const auto error = hipMemcpyAsync(
              device_auxiliary, impl->artifact.auxiliary(tensor).data(), tensor.aux_bytes,
              hipMemcpyHostToDevice, impl->stream);
          error != hipSuccess) {
        return hip_status(error, "scale upload failed for " + tensor.name);
      }
    }
    impl->device_tensors.push_back({
        tensor.name,
        tensor.dtype,
        tensor.quantization,
        tensor.rank,
        tensor.dimensions,
        device_data,
        device_auxiliary,
        tensor.data_bytes,
        tensor.aux_bytes,
    });
  }
  if (const auto error = hipStreamSynchronize(impl->stream); error != hipSuccess) {
    return hip_status(error, "resident weight upload synchronization failed");
  }
  std::size_t device_free_after = 0;
  if (const auto error = hipMemGetInfo(&device_free_after, &device_total);
      error != hipSuccess) {
    return hip_status(error, "hipMemGetInfo after weight load failed");
  }
  impl->summary.device_free_bytes_after_load = device_free_after;
  return Engine(std::move(impl));
}

const LoadSummary& Engine::load_summary() const noexcept { return impl_->summary; }

const DeviceTensorView* Engine::find_device_tensor(std::string_view name) const noexcept {
  const auto found = std::lower_bound(
      impl_->device_tensors.begin(), impl_->device_tensors.end(), name,
      [](const auto& tensor, std::string_view target) { return tensor.name < target; });
  return found != impl_->device_tensors.end() && found->name == name ? &*found : nullptr;
}

hipStream_t Engine::compute_stream() const noexcept { return impl_->stream; }

std::span<const std::byte> Engine::resource(std::string_view name) const noexcept {
  const auto* tensor = impl_->artifact.find(name);
  if (tensor == nullptr || !is_resource(tensor->name) ||
      tensor->quantization != Quantization::none || tensor->dtype != DType::u8) {
    return {};
  }
  return impl_->artifact.data(*tensor);
}

}  // namespace gfxinfer
