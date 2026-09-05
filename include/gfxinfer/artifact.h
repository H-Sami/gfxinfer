// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gfxinfer/status.h"

namespace gfxinfer {

enum class DType : std::uint16_t {
  u8 = 1,
  i32 = 2,
  f16 = 3,
  bf16 = 4,
  f32 = 5,
};

enum class Quantization : std::uint16_t {
  none = 0,
  w2g128 = 1,
  w3g128 = 2,
  w4g128 = 3,
  w8g128 = 4,
  fp8_row = 5,
};

struct ArtifactIdentity {
  std::string target_arch;
  std::string model_id;
  std::string weights_id;
  std::string recipe_id;
};

struct TensorDescriptor {
  std::string name;
  DType dtype{DType::u8};
  Quantization quantization{Quantization::none};
  std::uint16_t rank{0};
  std::uint16_t flags{0};
  std::uint32_t alignment{256};
  std::array<std::uint64_t, 4> dimensions{};
  std::uint64_t data_offset{0};
  std::uint64_t data_bytes{0};
  std::uint64_t aux_offset{0};
  std::uint64_t aux_bytes{0};
  std::uint64_t data_crc64{0};
};

struct TensorWriteRequest {
  std::string name;
  DType dtype{DType::u8};
  Quantization quantization{Quantization::none};
  std::vector<std::uint64_t> dimensions;
  std::span<const std::byte> data;
  std::span<const std::byte> auxiliary;
  std::uint32_t alignment{256};
};

struct TensorWritePlan {
  std::string name;
  DType dtype{DType::u8};
  Quantization quantization{Quantization::none};
  std::vector<std::uint64_t> dimensions;
  std::uint64_t data_bytes{0};
  std::uint64_t auxiliary_bytes{0};
  std::uint32_t alignment{256};
};

struct ArtifactOpenOptions {
  bool verify_checksums{true};
};

class ArtifactView {
 public:
  ArtifactView() = default;
  ~ArtifactView();
  ArtifactView(const ArtifactView&) = delete;
  ArtifactView& operator=(const ArtifactView&) = delete;
  ArtifactView(ArtifactView&& other) noexcept;
  ArtifactView& operator=(ArtifactView&& other) noexcept;

  [[nodiscard]] static Result<ArtifactView> open(
      const std::filesystem::path& path,
      ArtifactOpenOptions options = {});

  [[nodiscard]] const ArtifactIdentity& identity() const noexcept { return identity_; }
  [[nodiscard]] const std::vector<TensorDescriptor>& tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] const TensorDescriptor* find(std::string_view name) const noexcept;
  [[nodiscard]] std::span<const std::byte> data(const TensorDescriptor& tensor) const;
  [[nodiscard]] std::span<const std::byte> auxiliary(
      const TensorDescriptor& tensor) const;
  [[nodiscard]] std::uint64_t file_bytes() const noexcept {
    return static_cast<std::uint64_t>(mapped_bytes_);
  }

 private:
  void reset() noexcept;

  int fd_{-1};
  const std::byte* mapping_{nullptr};
  std::size_t mapped_bytes_{0};
  ArtifactIdentity identity_;
  std::vector<TensorDescriptor> tensors_;
};

class ArtifactWriter {
 public:
  [[nodiscard]] static Status write(
      const std::filesystem::path& path,
      const ArtifactIdentity& identity,
      std::span<const TensorWriteRequest> tensors);
};

class ArtifactStreamWriter {
 public:
  ArtifactStreamWriter();
  ~ArtifactStreamWriter();
  ArtifactStreamWriter(const ArtifactStreamWriter&) = delete;
  ArtifactStreamWriter& operator=(const ArtifactStreamWriter&) = delete;
  ArtifactStreamWriter(ArtifactStreamWriter&&) noexcept;
  ArtifactStreamWriter& operator=(ArtifactStreamWriter&&) noexcept;

  [[nodiscard]] static Result<ArtifactStreamWriter> create(
      const std::filesystem::path& path,
      const ArtifactIdentity& identity,
      std::span<const TensorWritePlan> tensors);

  [[nodiscard]] Status append_data(
      std::size_t tensor_index,
      std::span<const std::byte> bytes);
  [[nodiscard]] Status append_auxiliary(
      std::size_t tensor_index,
      std::span<const std::byte> bytes);
  [[nodiscard]] Status finalize();

 private:
  struct Impl;
  explicit ArtifactStreamWriter(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::uint64_t crc64_ecma(std::span<const std::byte> bytes) noexcept;
[[nodiscard]] std::uint64_t crc64_ecma_update(
    std::uint64_t state,
    std::span<const std::byte> bytes) noexcept;
[[nodiscard]] const char* dtype_name(DType dtype) noexcept;
[[nodiscard]] const char* quantization_name(Quantization quantization) noexcept;

}  // namespace gfxinfer
