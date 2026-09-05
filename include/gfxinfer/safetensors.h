// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gfxinfer/status.h"

namespace gfxinfer {

enum class SourceDType {
  bf16,
  f16,
  f32,
  u8,
  i8,
  i32,
  i64,
};

struct SafeTensorDescriptor {
  std::string name;
  SourceDType dtype{SourceDType::bf16};
  std::vector<std::uint64_t> shape;
  std::uint64_t file_offset{0};
  std::uint64_t bytes{0};
};

class SafeTensorFile {
 public:
  SafeTensorFile() = default;
  ~SafeTensorFile();
  SafeTensorFile(const SafeTensorFile&) = delete;
  SafeTensorFile& operator=(const SafeTensorFile&) = delete;
  SafeTensorFile(SafeTensorFile&& other) noexcept;
  SafeTensorFile& operator=(SafeTensorFile&& other) noexcept;

  [[nodiscard]] static Result<SafeTensorFile> open(const std::filesystem::path& path);
  [[nodiscard]] const std::vector<SafeTensorDescriptor>& tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] const SafeTensorDescriptor* find(std::string_view name) const noexcept;
  [[nodiscard]] std::span<const std::byte> data(const SafeTensorDescriptor& tensor) const;
  [[nodiscard]] std::uint64_t file_bytes() const noexcept {
    return static_cast<std::uint64_t>(mapped_bytes_);
  }

 private:
  void reset() noexcept;

  int fd_{-1};
  const std::byte* mapping_{nullptr};
  std::size_t mapped_bytes_{0};
  std::vector<SafeTensorDescriptor> tensors_;
};

struct SafeTensorIndex {
  std::uint64_t tensor_bytes{0};
  std::map<std::string, std::string> weight_map;
  std::vector<std::string> shards;
};

struct SourceValidationSummary {
  std::size_t tensor_count{0};
  std::size_t shard_count{0};
  std::uint64_t tensor_bytes{0};
};

[[nodiscard]] Result<SafeTensorIndex> load_safetensors_index(
    const std::filesystem::path& path);

[[nodiscard]] Result<SourceValidationSummary> validate_qwen38_model_directory(
    const std::filesystem::path& directory,
    bool include_vision = true,
    bool include_mtp = true);

[[nodiscard]] const char* source_dtype_name(SourceDType dtype) noexcept;
[[nodiscard]] std::size_t source_dtype_bytes(SourceDType dtype) noexcept;

}  // namespace gfxinfer
