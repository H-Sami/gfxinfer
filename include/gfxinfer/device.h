// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gfxinfer/status.h"

namespace gfxinfer {

struct DeviceInfo {
  int index{-1};
  std::string name;
  std::string architecture;
  std::size_t total_memory_bytes{0};
  int compute_units{0};
  int clock_khz{0};
  int memory_clock_khz{0};
  int memory_bus_width_bits{0};
  int warp_size{0};
};

[[nodiscard]] Result<std::vector<DeviceInfo>> enumerate_devices();
[[nodiscard]] Result<DeviceInfo> require_gfx1200_device(int index);
[[nodiscard]] Status hip_status(int error_code, std::string context);

}  // namespace gfxinfer
