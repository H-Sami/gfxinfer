// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/artifact.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gfxinfer/version.h"

namespace gfxinfer {
namespace {

constexpr std::array<char, 8> kMagic{'G', 'F', 'X', 'I', '\r', '\n', 0x1a, '\n'};
constexpr std::uint32_t kEndianTag = 0x01020304U;
constexpr std::size_t kFileAlignment = 256;

#pragma pack(push, 1)
struct DiskHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint32_t endian_tag;
  std::uint32_t flags;
  std::uint64_t directory_offset;
  std::uint32_t tensor_count;
  std::uint32_t directory_entry_bytes;
  std::uint64_t strings_offset;
  std::uint64_t strings_bytes;
  std::uint64_t data_offset;
  std::uint64_t data_bytes;
  std::uint64_t file_bytes;
  std::uint64_t manifest_crc64;
  char target_arch[16];
  char model_id[32];
  char weights_id[32];
  char recipe_id[64];
  std::uint8_t reserved[24];
};

struct DiskTensorEntry {
  std::uint64_t name_offset;
  std::uint32_t name_bytes;
  std::uint16_t dtype;
  std::uint16_t quantization;
  std::uint16_t rank;
  std::uint16_t flags;
  std::uint32_t alignment;
  std::uint64_t dimensions[4];
  std::uint64_t data_offset;
  std::uint64_t data_bytes;
  std::uint64_t aux_offset;
  std::uint64_t aux_bytes;
  std::uint64_t data_crc64;
  std::uint8_t reserved[32];
};
#pragma pack(pop)

static_assert(sizeof(DiskHeader) == 256);
static_assert(sizeof(DiskTensorEntry) == 128);

[[nodiscard]] std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return 0;
  }
  const auto mask = alignment - 1;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return 0;
  }
  return (value + mask) & ~mask;
}

[[nodiscard]] bool range_valid(
    std::uint64_t offset,
    std::uint64_t bytes,
    std::uint64_t file_bytes) {
  return offset <= file_bytes && bytes <= file_bytes - offset;
}

template <std::size_t N>
[[nodiscard]] std::string fixed_string(const char (&value)[N]) {
  const auto* end = static_cast<const char*>(std::memchr(value, '\0', N));
  return std::string(value, end == nullptr ? value + N : end);
}

template <std::size_t N>
[[nodiscard]] bool set_fixed(char (&destination)[N], const std::string& value) {
  if (value.empty() || value.size() >= N) {
    return false;
  }
  std::memset(destination, 0, N);
  std::memcpy(destination, value.data(), value.size());
  return true;
}

[[nodiscard]] bool valid_name(std::string_view name) {
  if (name.empty() || name.front() == '/' || name.find("..") != std::string_view::npos) {
    return false;
  }
  return std::all_of(name.begin(), name.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' || c == '/';
  });
}

[[nodiscard]] bool valid_dtype(std::uint16_t value) {
  return value >= static_cast<std::uint16_t>(DType::u8) &&
         value <= static_cast<std::uint16_t>(DType::f32);
}

[[nodiscard]] bool valid_quantization(std::uint16_t value) {
  return value <= static_cast<std::uint16_t>(Quantization::fp8_row);
}

[[nodiscard]] Status errno_status(std::string_view action, const std::filesystem::path& path) {
  return {ErrorCode::io_error,
          std::string(action) + " " + path.string() + ": " + std::strerror(errno)};
}

[[nodiscard]] Status write_all(int fd, const void* source, std::size_t bytes) {
  const auto* cursor = static_cast<const std::byte*>(source);
  while (bytes != 0) {
    const auto written = ::write(fd, cursor, bytes);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return {ErrorCode::io_error, std::string("write failed: ") + std::strerror(errno)};
    }
    if (written == 0) {
      return {ErrorCode::io_error, "write returned zero before completion"};
    }
    cursor += written;
    bytes -= static_cast<std::size_t>(written);
  }
  return Status::ok_status();
}

[[nodiscard]] Status pwrite_all(
    int fd,
    const void* source,
    std::size_t bytes,
    std::uint64_t offset) {
  const auto* cursor = static_cast<const std::byte*>(source);
  while (bytes != 0) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      return {ErrorCode::io_error, "file offset exceeds platform limit"};
    }
    const auto written = ::pwrite(fd, cursor, bytes, static_cast<off_t>(offset));
    if (written < 0) {
      if (errno == EINTR) continue;
      return {ErrorCode::io_error, std::string("pwrite failed: ") + std::strerror(errno)};
    }
    if (written == 0) {
      return {ErrorCode::io_error, "pwrite returned zero before completion"};
    }
    cursor += written;
    bytes -= static_cast<std::size_t>(written);
    offset += static_cast<std::uint64_t>(written);
  }
  return Status::ok_status();
}

[[nodiscard]] Status write_padding(int fd, std::uint64_t bytes) {
  static constexpr std::array<std::byte, kFileAlignment> zeros{};
  while (bytes != 0) {
    const auto chunk = static_cast<std::size_t>(
        std::min<std::uint64_t>(bytes, zeros.size()));
    auto status = write_all(fd, zeros.data(), chunk);
    if (!status.ok()) {
      return status;
    }
    bytes -= chunk;
  }
  return Status::ok_status();
}

}  // namespace

std::uint64_t crc64_ecma(std::span<const std::byte> bytes) noexcept {
  return crc64_ecma_update(0, bytes);
}

std::uint64_t crc64_ecma_update(
    std::uint64_t crc,
    std::span<const std::byte> bytes) noexcept {
  constexpr std::uint64_t polynomial = 0x42F0E1EBA9EA3693ULL;
  static constexpr auto table = [] {
    std::array<std::uint64_t, 256> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      std::uint64_t value = static_cast<std::uint64_t>(index) << 56U;
      for (int bit = 0; bit < 8; ++bit) {
        value = (value & (1ULL << 63U)) != 0 ? (value << 1U) ^ polynomial
                                             : value << 1U;
      }
      values[index] = value;
    }
    return values;
  }();
  for (const auto byte : bytes) {
    const auto index = static_cast<std::uint8_t>(
        (crc >> 56U) ^ std::to_integer<std::uint8_t>(byte));
    crc = table[index] ^ (crc << 8U);
  }
  return crc;
}

const char* dtype_name(DType dtype) noexcept {
  switch (dtype) {
    case DType::u8: return "u8";
    case DType::i32: return "i32";
    case DType::f16: return "f16";
    case DType::bf16: return "bf16";
    case DType::f32: return "f32";
  }
  return "unknown";
}

const char* quantization_name(Quantization quantization) noexcept {
  switch (quantization) {
    case Quantization::none: return "none";
    case Quantization::w2g128: return "w2g128";
    case Quantization::w3g128: return "w3g128";
    case Quantization::w4g128: return "w4g128";
    case Quantization::w8g128: return "w8g128";
    case Quantization::fp8_row: return "fp8-row";
  }
  return "unknown";
}

ArtifactView::~ArtifactView() { reset(); }

ArtifactView::ArtifactView(ArtifactView&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      mapped_bytes_(std::exchange(other.mapped_bytes_, 0)),
      identity_(std::move(other.identity_)),
      tensors_(std::move(other.tensors_)) {}

ArtifactView& ArtifactView::operator=(ArtifactView&& other) noexcept {
  if (this != &other) {
    reset();
    fd_ = std::exchange(other.fd_, -1);
    mapping_ = std::exchange(other.mapping_, nullptr);
    mapped_bytes_ = std::exchange(other.mapped_bytes_, 0);
    identity_ = std::move(other.identity_);
    tensors_ = std::move(other.tensors_);
  }
  return *this;
}

void ArtifactView::reset() noexcept {
  if (mapping_ != nullptr) {
    ::munmap(const_cast<std::byte*>(mapping_), mapped_bytes_);
  }
  if (fd_ >= 0) {
    ::close(fd_);
  }
  fd_ = -1;
  mapping_ = nullptr;
  mapped_bytes_ = 0;
  identity_ = {};
  tensors_.clear();
}

Result<ArtifactView> ArtifactView::open(
    const std::filesystem::path& path,
    ArtifactOpenOptions options) {
  ArtifactView view;
  view.fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (view.fd_ < 0) {
    return errno_status("cannot open", path);
  }

  struct stat file_stat {};
  if (::fstat(view.fd_, &file_stat) != 0) {
    return errno_status("cannot stat", path);
  }
  if (file_stat.st_size < static_cast<off_t>(sizeof(DiskHeader))) {
    return Status{ErrorCode::corrupt_artifact, "artifact is smaller than its header"};
  }
  view.mapped_bytes_ = static_cast<std::size_t>(file_stat.st_size);
  const auto* mapping = ::mmap(nullptr, view.mapped_bytes_, PROT_READ, MAP_PRIVATE, view.fd_, 0);
  if (mapping == MAP_FAILED) {
    view.mapping_ = nullptr;
    return errno_status("cannot map", path);
  }
  view.mapping_ = static_cast<const std::byte*>(mapping);

  const auto* header = reinterpret_cast<const DiskHeader*>(view.mapping_);
  if (!std::equal(kMagic.begin(), kMagic.end(), header->magic)) {
    return Status{ErrorCode::corrupt_artifact, "invalid .gfxi magic"};
  }
  if (header->version != kArtifactVersion || header->header_bytes != sizeof(DiskHeader)) {
    return Status{ErrorCode::unsupported, "unsupported .gfxi artifact version"};
  }
  if (header->endian_tag != kEndianTag) {
    return Status{ErrorCode::unsupported, "artifact byte order is unsupported"};
  }
  if (header->directory_entry_bytes != sizeof(DiskTensorEntry)) {
    return Status{ErrorCode::corrupt_artifact, "unexpected directory entry size"};
  }
  if (header->file_bytes != view.mapped_bytes_) {
    return Status{ErrorCode::corrupt_artifact, "artifact length does not match header"};
  }

  const auto directory_bytes =
      static_cast<std::uint64_t>(header->tensor_count) * sizeof(DiskTensorEntry);
  if (!range_valid(header->directory_offset, directory_bytes, header->file_bytes) ||
      !range_valid(header->strings_offset, header->strings_bytes, header->file_bytes) ||
      !range_valid(header->data_offset, header->data_bytes, header->file_bytes) ||
      header->strings_offset != header->directory_offset + directory_bytes ||
      header->data_offset < header->strings_offset + header->strings_bytes) {
    return Status{ErrorCode::corrupt_artifact, "artifact section bounds are invalid"};
  }

  const auto manifest_bytes = directory_bytes + header->strings_bytes;
  if (options.verify_checksums &&
      crc64_ecma({view.mapping_ + header->directory_offset,
                  static_cast<std::size_t>(manifest_bytes)}) != header->manifest_crc64) {
    return Status{ErrorCode::corrupt_artifact, "artifact manifest checksum mismatch"};
  }

  view.identity_ = {
      fixed_string(header->target_arch),
      fixed_string(header->model_id),
      fixed_string(header->weights_id),
      fixed_string(header->recipe_id),
  };
  if (view.identity_.target_arch.empty() || view.identity_.model_id.empty() ||
      view.identity_.weights_id.empty() || view.identity_.recipe_id.empty()) {
    return Status{ErrorCode::corrupt_artifact, "artifact identity is incomplete"};
  }

  const auto* entries = reinterpret_cast<const DiskTensorEntry*>(
      view.mapping_ + header->directory_offset);
  const auto* strings = reinterpret_cast<const char*>(view.mapping_ + header->strings_offset);
  std::set<std::string> unique_names;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  view.tensors_.reserve(header->tensor_count);
  for (std::uint32_t index = 0; index < header->tensor_count; ++index) {
    const auto& entry = entries[index];
    if (entry.rank > 4 || !valid_dtype(entry.dtype) ||
        !valid_quantization(entry.quantization) || entry.name_bytes == 0 ||
        !range_valid(entry.name_offset, entry.name_bytes, header->strings_bytes)) {
      return Status{ErrorCode::corrupt_artifact, "invalid tensor directory entry"};
    }
    const std::string name(strings + entry.name_offset, entry.name_bytes);
    if (!valid_name(name) || !unique_names.insert(name).second) {
      return Status{ErrorCode::corrupt_artifact, "invalid or duplicate tensor name: " + name};
    }
    if (entry.alignment < kFileAlignment ||
        (entry.alignment & (entry.alignment - 1)) != 0 ||
        entry.data_offset % entry.alignment != 0 ||
        !range_valid(entry.data_offset, entry.data_bytes, header->file_bytes) ||
        entry.data_offset < header->data_offset) {
      return Status{ErrorCode::corrupt_artifact, "invalid tensor payload bounds: " + name};
    }
    if (entry.aux_bytes != 0 &&
        (entry.aux_offset % entry.alignment != 0 ||
         !range_valid(entry.aux_offset, entry.aux_bytes, header->file_bytes) ||
         entry.aux_offset < header->data_offset)) {
      return Status{ErrorCode::corrupt_artifact, "invalid tensor auxiliary bounds: " + name};
    }
    if (options.verify_checksums &&
        crc64_ecma({view.mapping_ + entry.data_offset,
                    static_cast<std::size_t>(entry.data_bytes)}) != entry.data_crc64) {
      return Status{ErrorCode::corrupt_artifact, "tensor checksum mismatch: " + name};
    }

    TensorDescriptor descriptor;
    descriptor.name = name;
    descriptor.dtype = static_cast<DType>(entry.dtype);
    descriptor.quantization = static_cast<Quantization>(entry.quantization);
    descriptor.rank = entry.rank;
    descriptor.flags = entry.flags;
    descriptor.alignment = entry.alignment;
    std::copy(std::begin(entry.dimensions), std::end(entry.dimensions),
              descriptor.dimensions.begin());
    descriptor.data_offset = entry.data_offset;
    descriptor.data_bytes = entry.data_bytes;
    descriptor.aux_offset = entry.aux_offset;
    descriptor.aux_bytes = entry.aux_bytes;
    descriptor.data_crc64 = entry.data_crc64;
    view.tensors_.push_back(std::move(descriptor));
    ranges.emplace_back(entry.data_offset, entry.data_offset + entry.data_bytes);
    if (entry.aux_bytes != 0) {
      ranges.emplace_back(entry.aux_offset, entry.aux_offset + entry.aux_bytes);
    }
  }

  std::sort(ranges.begin(), ranges.end());
  for (std::size_t index = 1; index < ranges.size(); ++index) {
    if (ranges[index].first < ranges[index - 1].second) {
      return Status{ErrorCode::corrupt_artifact, "tensor payload regions overlap"};
    }
  }
  return Result<ArtifactView>(std::move(view));
}

const TensorDescriptor* ArtifactView::find(std::string_view name) const noexcept {
  const auto found = std::find_if(tensors_.begin(), tensors_.end(),
                                  [name](const auto& tensor) { return tensor.name == name; });
  return found == tensors_.end() ? nullptr : &*found;
}

std::span<const std::byte> ArtifactView::data(const TensorDescriptor& tensor) const {
  return {mapping_ + tensor.data_offset, static_cast<std::size_t>(tensor.data_bytes)};
}

std::span<const std::byte> ArtifactView::auxiliary(const TensorDescriptor& tensor) const {
  if (tensor.aux_bytes == 0) {
    return {};
  }
  return {mapping_ + tensor.aux_offset, static_cast<std::size_t>(tensor.aux_bytes)};
}

Status ArtifactWriter::write(
    const std::filesystem::path& path,
    const ArtifactIdentity& identity,
    std::span<const TensorWriteRequest> tensors) {
  if (tensors.size() > std::numeric_limits<std::uint32_t>::max()) {
    return {ErrorCode::invalid_argument, "too many tensors for .gfxi v1"};
  }

  DiskHeader header{};
  std::copy(kMagic.begin(), kMagic.end(), header.magic);
  header.version = kArtifactVersion;
  header.header_bytes = sizeof(DiskHeader);
  header.endian_tag = kEndianTag;
  header.directory_offset = sizeof(DiskHeader);
  header.tensor_count = static_cast<std::uint32_t>(tensors.size());
  header.directory_entry_bytes = sizeof(DiskTensorEntry);
  if (!set_fixed(header.target_arch, identity.target_arch) ||
      !set_fixed(header.model_id, identity.model_id) ||
      !set_fixed(header.weights_id, identity.weights_id) ||
      !set_fixed(header.recipe_id, identity.recipe_id)) {
    return {ErrorCode::invalid_argument, "artifact identity is empty or too long"};
  }

  std::set<std::string> unique_names;
  std::vector<DiskTensorEntry> entries(tensors.size());
  std::vector<std::byte> strings;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto& request = tensors[index];
    if (!valid_name(request.name) || !unique_names.insert(request.name).second ||
        request.dimensions.size() > 4 || request.data.empty() ||
        request.alignment < kFileAlignment ||
        (request.alignment & (request.alignment - 1)) != 0) {
      return {ErrorCode::invalid_argument, "invalid tensor write request: " + request.name};
    }
    auto& entry = entries[index];
    entry.name_offset = strings.size();
    entry.name_bytes = static_cast<std::uint32_t>(request.name.size());
    entry.dtype = static_cast<std::uint16_t>(request.dtype);
    entry.quantization = static_cast<std::uint16_t>(request.quantization);
    entry.rank = static_cast<std::uint16_t>(request.dimensions.size());
    entry.alignment = request.alignment;
    std::copy(request.dimensions.begin(), request.dimensions.end(), entry.dimensions);
    entry.data_bytes = request.data.size();
    entry.aux_bytes = request.auxiliary.size();
    entry.data_crc64 = crc64_ecma(request.data);
    const auto* name_bytes = reinterpret_cast<const std::byte*>(request.name.data());
    strings.insert(strings.end(), name_bytes, name_bytes + request.name.size());
  }

  const auto directory_bytes = entries.size() * sizeof(DiskTensorEntry);
  header.strings_offset = header.directory_offset + directory_bytes;
  header.strings_bytes = strings.size();
  header.data_offset = align_up(header.strings_offset + header.strings_bytes, kFileAlignment);
  if (header.data_offset == 0) {
    return {ErrorCode::invalid_argument, "artifact size overflow"};
  }

  std::uint64_t cursor = header.data_offset;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    auto& entry = entries[index];
    cursor = align_up(cursor, entry.alignment);
    if (cursor == 0) {
      return {ErrorCode::invalid_argument, "artifact payload offset overflow"};
    }
    entry.data_offset = cursor;
    cursor += entry.data_bytes;
    if (entry.aux_bytes != 0) {
      cursor = align_up(cursor, entry.alignment);
      if (cursor == 0) {
        return {ErrorCode::invalid_argument, "artifact auxiliary offset overflow"};
      }
      entry.aux_offset = cursor;
      cursor += entry.aux_bytes;
    }
  }
  header.file_bytes = cursor;
  header.data_bytes = cursor - header.data_offset;

  std::vector<std::byte> manifest(directory_bytes + strings.size());
  if (!entries.empty()) {
    std::memcpy(manifest.data(), entries.data(), directory_bytes);
  }
  if (!strings.empty()) {
    std::memcpy(manifest.data() + directory_bytes, strings.data(), strings.size());
  }
  header.manifest_crc64 = crc64_ecma(manifest);

  const auto temporary = path.string() + ".tmp." + std::to_string(::getpid());
  const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) {
    return errno_status("cannot create", temporary);
  }
  bool success = false;
  auto finish = [&](Status status) {
    const int saved_errno = errno;
    ::close(fd);
    if (!success) {
      ::unlink(temporary.c_str());
    }
    errno = saved_errno;
    return status;
  };

  auto status = write_all(fd, &header, sizeof(header));
  if (!status.ok()) return finish(status);
  status = write_all(fd, manifest.data(), manifest.size());
  if (!status.ok()) return finish(status);
  const auto manifest_end = header.strings_offset + header.strings_bytes;
  status = write_padding(fd, header.data_offset - manifest_end);
  if (!status.ok()) return finish(status);

  cursor = header.data_offset;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto& request = tensors[index];
    const auto& entry = entries[index];
    status = write_padding(fd, entry.data_offset - cursor);
    if (!status.ok()) return finish(status);
    status = write_all(fd, request.data.data(), request.data.size());
    if (!status.ok()) return finish(status);
    cursor = entry.data_offset + entry.data_bytes;
    if (entry.aux_bytes != 0) {
      status = write_padding(fd, entry.aux_offset - cursor);
      if (!status.ok()) return finish(status);
      status = write_all(fd, request.auxiliary.data(), request.auxiliary.size());
      if (!status.ok()) return finish(status);
      cursor = entry.aux_offset + entry.aux_bytes;
    }
  }
  if (::fsync(fd) != 0) {
    return finish({ErrorCode::io_error, std::string("fsync failed: ") + std::strerror(errno)});
  }
  if (::rename(temporary.c_str(), path.c_str()) != 0) {
    return finish(errno_status("cannot install artifact", path));
  }
  success = true;
  return finish(Status::ok_status());
}

struct ArtifactStreamWriter::Impl {
  struct TensorState {
    std::uint64_t data_written{0};
    std::uint64_t auxiliary_written{0};
    std::uint64_t data_crc64{0};
  };

  ~Impl() {
    if (fd >= 0) (void)::close(fd);
    if (!finalized && !temporary_path.empty()) (void)::unlink(temporary_path.c_str());
  }

  std::filesystem::path final_path;
  std::string temporary_path;
  int fd{-1};
  DiskHeader header{};
  std::vector<DiskTensorEntry> entries;
  std::vector<std::byte> strings;
  std::vector<TensorState> states;
  bool finalized{false};
};

ArtifactStreamWriter::ArtifactStreamWriter() = default;
ArtifactStreamWriter::~ArtifactStreamWriter() = default;
ArtifactStreamWriter::ArtifactStreamWriter(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ArtifactStreamWriter::ArtifactStreamWriter(ArtifactStreamWriter&&) noexcept = default;
ArtifactStreamWriter& ArtifactStreamWriter::operator=(ArtifactStreamWriter&&) noexcept = default;

Result<ArtifactStreamWriter> ArtifactStreamWriter::create(
    const std::filesystem::path& path,
    const ArtifactIdentity& identity,
    std::span<const TensorWritePlan> tensors) {
  if (tensors.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status{ErrorCode::invalid_argument, "too many tensors for .gfxi v1"};
  }
  auto impl = std::make_unique<Impl>();
  impl->final_path = path;
  impl->temporary_path = path.string() + ".tmp." + std::to_string(::getpid());
  impl->entries.resize(tensors.size());
  impl->states.resize(tensors.size());

  auto& header = impl->header;
  std::copy(kMagic.begin(), kMagic.end(), header.magic);
  header.version = kArtifactVersion;
  header.header_bytes = sizeof(DiskHeader);
  header.endian_tag = kEndianTag;
  header.directory_offset = sizeof(DiskHeader);
  header.tensor_count = static_cast<std::uint32_t>(tensors.size());
  header.directory_entry_bytes = sizeof(DiskTensorEntry);
  if (!set_fixed(header.target_arch, identity.target_arch) ||
      !set_fixed(header.model_id, identity.model_id) ||
      !set_fixed(header.weights_id, identity.weights_id) ||
      !set_fixed(header.recipe_id, identity.recipe_id)) {
    return Status{ErrorCode::invalid_argument, "artifact identity is empty or too long"};
  }

  std::set<std::string> unique_names;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto& plan = tensors[index];
    if (!valid_name(plan.name) || !unique_names.insert(plan.name).second ||
        plan.dimensions.size() > 4 || plan.data_bytes == 0 ||
        plan.alignment < kFileAlignment ||
        (plan.alignment & (plan.alignment - 1)) != 0) {
      return Status{ErrorCode::invalid_argument, "invalid tensor stream plan: " + plan.name};
    }
    auto& entry = impl->entries[index];
    entry.name_offset = impl->strings.size();
    entry.name_bytes = static_cast<std::uint32_t>(plan.name.size());
    entry.dtype = static_cast<std::uint16_t>(plan.dtype);
    entry.quantization = static_cast<std::uint16_t>(plan.quantization);
    entry.rank = static_cast<std::uint16_t>(plan.dimensions.size());
    entry.alignment = plan.alignment;
    std::copy(plan.dimensions.begin(), plan.dimensions.end(), entry.dimensions);
    entry.data_bytes = plan.data_bytes;
    entry.aux_bytes = plan.auxiliary_bytes;
    const auto* name_bytes = reinterpret_cast<const std::byte*>(plan.name.data());
    impl->strings.insert(impl->strings.end(), name_bytes, name_bytes + plan.name.size());
  }

  const auto directory_bytes = impl->entries.size() * sizeof(DiskTensorEntry);
  header.strings_offset = header.directory_offset + directory_bytes;
  header.strings_bytes = impl->strings.size();
  header.data_offset = align_up(header.strings_offset + header.strings_bytes, kFileAlignment);
  if (header.data_offset == 0) {
    return Status{ErrorCode::invalid_argument, "artifact size overflow"};
  }
  std::uint64_t cursor = header.data_offset;
  for (auto& entry : impl->entries) {
    cursor = align_up(cursor, entry.alignment);
    if (cursor == 0 || entry.data_bytes > std::numeric_limits<std::uint64_t>::max() - cursor) {
      return Status{ErrorCode::invalid_argument, "artifact tensor offset overflow"};
    }
    entry.data_offset = cursor;
    cursor += entry.data_bytes;
    if (entry.aux_bytes != 0) {
      cursor = align_up(cursor, entry.alignment);
      if (cursor == 0 || entry.aux_bytes > std::numeric_limits<std::uint64_t>::max() - cursor) {
        return Status{ErrorCode::invalid_argument, "artifact auxiliary offset overflow"};
      }
      entry.aux_offset = cursor;
      cursor += entry.aux_bytes;
    }
  }
  header.file_bytes = cursor;
  header.data_bytes = cursor - header.data_offset;
  if (header.file_bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    return Status{ErrorCode::invalid_argument, "artifact exceeds platform file-size limit"};
  }

  impl->fd = ::open(impl->temporary_path.c_str(),
                    O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (impl->fd < 0) return errno_status("cannot create", impl->temporary_path);
  if (::ftruncate(impl->fd, static_cast<off_t>(header.file_bytes)) != 0) {
    return errno_status("cannot size", impl->temporary_path);
  }
  auto status = pwrite_all(impl->fd, &header, sizeof(header), 0);
  if (!status.ok()) return status;
  if (!impl->entries.empty()) {
    status = pwrite_all(impl->fd, impl->entries.data(), directory_bytes,
                        header.directory_offset);
    if (!status.ok()) return status;
  }
  if (!impl->strings.empty()) {
    status = pwrite_all(impl->fd, impl->strings.data(), impl->strings.size(),
                        header.strings_offset);
    if (!status.ok()) return status;
  }
  return ArtifactStreamWriter(std::move(impl));
}

Status ArtifactStreamWriter::append_data(
    std::size_t tensor_index,
    std::span<const std::byte> bytes) {
  if (!impl_ || impl_->finalized || tensor_index >= impl_->entries.size()) {
    return {ErrorCode::invalid_argument, "invalid or finalized artifact stream writer"};
  }
  auto& state = impl_->states[tensor_index];
  const auto& entry = impl_->entries[tensor_index];
  if (bytes.size() > entry.data_bytes - state.data_written) {
    return {ErrorCode::invalid_argument, "tensor data stream exceeds its declared size"};
  }
  auto status = pwrite_all(impl_->fd, bytes.data(), bytes.size(),
                           entry.data_offset + state.data_written);
  if (!status.ok()) return status;
  state.data_crc64 = crc64_ecma_update(state.data_crc64, bytes);
  state.data_written += bytes.size();
  return Status::ok_status();
}

Status ArtifactStreamWriter::append_auxiliary(
    std::size_t tensor_index,
    std::span<const std::byte> bytes) {
  if (!impl_ || impl_->finalized || tensor_index >= impl_->entries.size()) {
    return {ErrorCode::invalid_argument, "invalid or finalized artifact stream writer"};
  }
  auto& state = impl_->states[tensor_index];
  const auto& entry = impl_->entries[tensor_index];
  if (bytes.size() > entry.aux_bytes - state.auxiliary_written) {
    return {ErrorCode::invalid_argument, "tensor auxiliary stream exceeds its declared size"};
  }
  auto status = pwrite_all(impl_->fd, bytes.data(), bytes.size(),
                           entry.aux_offset + state.auxiliary_written);
  if (!status.ok()) return status;
  state.auxiliary_written += bytes.size();
  return Status::ok_status();
}

Status ArtifactStreamWriter::finalize() {
  if (!impl_ || impl_->finalized) {
    return {ErrorCode::invalid_argument, "artifact stream writer is absent or already finalized"};
  }
  for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
    auto& entry = impl_->entries[index];
    const auto& state = impl_->states[index];
    if (state.data_written != entry.data_bytes ||
        state.auxiliary_written != entry.aux_bytes) {
      return {ErrorCode::invalid_argument,
              "artifact stream is incomplete at tensor index " + std::to_string(index)};
    }
    entry.data_crc64 = state.data_crc64;
  }

  const auto directory_bytes = impl_->entries.size() * sizeof(DiskTensorEntry);
  std::vector<std::byte> manifest(directory_bytes + impl_->strings.size());
  if (!impl_->entries.empty()) {
    std::memcpy(manifest.data(), impl_->entries.data(), directory_bytes);
  }
  if (!impl_->strings.empty()) {
    std::memcpy(manifest.data() + directory_bytes,
                impl_->strings.data(), impl_->strings.size());
  }
  impl_->header.manifest_crc64 = crc64_ecma(manifest);
  auto status = pwrite_all(impl_->fd, &impl_->header, sizeof(impl_->header), 0);
  if (!status.ok()) return status;
  status = pwrite_all(impl_->fd, manifest.data(), manifest.size(),
                      impl_->header.directory_offset);
  if (!status.ok()) return status;
  if (::fsync(impl_->fd) != 0) {
    return {ErrorCode::io_error, std::string("fsync failed: ") + std::strerror(errno)};
  }
  if (::close(impl_->fd) != 0) {
    impl_->fd = -1;
    return {ErrorCode::io_error, std::string("close failed: ") + std::strerror(errno)};
  }
  impl_->fd = -1;
  if (::rename(impl_->temporary_path.c_str(), impl_->final_path.c_str()) != 0) {
    return errno_status("cannot install artifact", impl_->final_path);
  }
  impl_->finalized = true;
  return Status::ok_status();
}

}  // namespace gfxinfer
