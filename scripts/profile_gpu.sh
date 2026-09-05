#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "usage: $0 OUTPUT.csv -- COMMAND [ARG ...]" >&2
  exit 2
fi

output="$1"
shift 2
device="/sys/class/drm/card0/device"
power_file="$(find "${device}/hwmon" -maxdepth 2 -name power1_average -print -quit)"
core_clock_file="$(find "${device}/hwmon" -maxdepth 2 -name freq1_input -print -quit)"
memory_clock_file="$(find "${device}/hwmon" -maxdepth 2 -name freq2_input -print -quit)"
if [[ ! -r "${device}/gpu_busy_percent" || ! -r "${device}/mem_busy_percent" ||
      ! -r "${power_file}" || ! -r "${core_clock_file}" ||
      ! -r "${memory_clock_file}" ]]; then
  echo "required amdgpu telemetry files are unavailable" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
printf 'timestamp_ns,gpu_busy_percent,mem_busy_percent,power_w,sclk_mhz,mclk_mhz\n' \
  > "${output}"

"$@" &
command_pid=$!
while kill -0 "${command_pid}" 2>/dev/null; do
  timestamp="$(date +%s%N)"
  gpu_busy="$(<"${device}/gpu_busy_percent")"
  mem_busy="$(<"${device}/mem_busy_percent")"
  power_uw="$(<"${power_file}")"
  sclk_hz="$(<"${core_clock_file}")"
  mclk_hz="$(<"${memory_clock_file}")"
  sclk="$((sclk_hz / 1000000))"
  mclk="$((mclk_hz / 1000000))"
  printf '%s,%s,%s,%.3f,%s,%s\n' \
    "${timestamp}" "${gpu_busy}" "${mem_busy}" "$((power_uw / 1000))e-3" \
    "${sclk:-0}" "${mclk:-0}" >> "${output}"
  sleep 0.05
done
wait "${command_pid}"
