// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "gfxinfer/artifact.h"
#include "gfxinfer/device.h"
#include "gfxinfer/status.h"

namespace gfxinfer {

enum class KvFormat { int8_group64, q4_group64 };

struct EngineConfig {
  std::filesystem::path artifact;
  int device_index{0};
  std::size_t memory_budget_bytes{0};
  std::uint32_t maximum_context_tokens{262144};
  std::uint32_t maximum_concurrency{8};
  KvFormat kv_format{KvFormat::int8_group64};
  std::uint32_t maximum_mtp_depth{7};
  bool verify_artifact_checksums{true};
  bool upload_weights{true};
};

struct LoadSummary {
  ArtifactIdentity identity;
  DeviceInfo device;
  std::uint64_t artifact_bytes{0};
  std::size_t configured_memory_budget_bytes{0};
  std::size_t tensor_count{0};
  std::size_t resident_tensor_count{0};
  std::size_t resident_weight_bytes{0};
  std::size_t device_free_bytes_before_load{0};
  std::size_t device_free_bytes_after_load{0};
};

struct DeviceTensorView {
  std::string name;
  DType dtype{DType::u8};
  Quantization quantization{Quantization::none};
  std::uint16_t rank{0};
  std::array<std::uint64_t, 4> dimensions{};
  const std::byte* data{nullptr};
  const std::byte* auxiliary{nullptr};
  std::uint64_t data_bytes{0};
  std::uint64_t auxiliary_bytes{0};
};

class PreparedPrompt {
 public:
  PreparedPrompt() = default;
  explicit PreparedPrompt(std::vector<std::uint32_t> tokens)
      : tokens_(std::move(tokens)) {}
  PreparedPrompt(const PreparedPrompt&) = delete;
  PreparedPrompt& operator=(const PreparedPrompt&) = delete;
  PreparedPrompt(PreparedPrompt&&) noexcept = default;
  PreparedPrompt& operator=(PreparedPrompt&&) noexcept = default;

  [[nodiscard]] const std::vector<std::uint32_t>& tokens() const noexcept {
    return tokens_;
  }

 private:
  std::vector<std::uint32_t> tokens_;
};

class Engine {
 public:
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) noexcept;
  Engine& operator=(Engine&&) noexcept;

  [[nodiscard]] static Result<Engine> create(EngineConfig config);
  [[nodiscard]] const LoadSummary& load_summary() const noexcept;
  [[nodiscard]] const DeviceTensorView* find_device_tensor(
      std::string_view name) const noexcept;
  [[nodiscard]] hipStream_t compute_stream() const noexcept;
  [[nodiscard]] std::span<const std::byte> resource(
      std::string_view name) const noexcept;

 private:
  struct Impl;
  explicit Engine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfxinfer
