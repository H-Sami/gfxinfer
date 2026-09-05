// SPDX-License-Identifier: Apache-2.0
#include "gfxinfer/device.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <string>

#include "gfxinfer/version.h"

namespace gfxinfer {
namespace {

[[nodiscard]] std::string base_architecture(const char* value) {
  std::string architecture(value == nullptr ? "" : value);
  if (const auto separator = architecture.find(':'); separator != std::string::npos) {
    architecture.resize(separator);
  }
  return architecture;
}

}  // namespace

Status hip_status(int error_code, std::string context) {
  const auto error = static_cast<hipError_t>(error_code);
  if (error == hipSuccess) {
    return Status::ok_status();
  }
  context += ": ";
  context += hipGetErrorName(error);
  context += " (";
  context += hipGetErrorString(error);
  context += ")";
  return {ErrorCode::hip_error, std::move(context)};
}

Result<std::vector<DeviceInfo>> enumerate_devices() {
  int count = 0;
  if (const auto error = hipGetDeviceCount(&count); error != hipSuccess) {
    return hip_status(error, "hipGetDeviceCount failed");
  }
  std::vector<DeviceInfo> devices;
  devices.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    hipDeviceProp_t properties{};
    if (const auto error = hipGetDeviceProperties(&properties, index); error != hipSuccess) {
      return hip_status(error, "hipGetDeviceProperties failed for device " +
                                   std::to_string(index));
    }
    devices.push_back({
        index,
        properties.name,
        base_architecture(properties.gcnArchName),
        properties.totalGlobalMem,
        properties.multiProcessorCount,
        properties.clockRate,
        properties.memoryClockRate,
        properties.memoryBusWidth,
        properties.warpSize,
    });
  }
  return devices;
}

Result<DeviceInfo> require_gfx1200_device(int index) {
  auto devices = enumerate_devices();
  if (!devices.ok()) {
    return devices.status();
  }
  const auto found = std::find_if(devices.value().begin(), devices.value().end(),
                                  [index](const auto& device) { return device.index == index; });
  if (found == devices.value().end()) {
    return Status{ErrorCode::invalid_argument,
                  "HIP device index " + std::to_string(index) + " does not exist"};
  }
  if (found->architecture != kRequiredArchitecture) {
    return Status{ErrorCode::unsupported,
                  "device " + std::to_string(index) + " is " + found->architecture +
                      "; GFXInfer requires gfx1200"};
  }
  if (found->warp_size != 32) {
    return Status{ErrorCode::unsupported,
                  "gfx1200 device reported unexpected wave size " +
                      std::to_string(found->warp_size)};
  }
  if (const auto error = hipSetDevice(index); error != hipSuccess) {
    return hip_status(error, "hipSetDevice failed");
  }
  return *found;
}

}  // namespace gfxinfer
