#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-${project_root}/models/ranking.train.counts.i64}"
partial="${destination}.partial"
revision="ad0f3d384b5cbcec4a48a3951c287b4e9831443e"
expected_sha256="c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5"
url="https://raw.githubusercontent.com/Neroued/ninfer/${revision}/tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"

mkdir -p "$(dirname "${destination}")"
curl --fail --location --continue-at - --limit-rate "${GFXINFER_DOWNLOAD_RATE:-16M}" \
  --output "${partial}" "${url}"
actual_sha256="$(sha256sum "${partial}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "ranking checksum mismatch: expected ${expected_sha256}, got ${actual_sha256}" >&2
  exit 1
fi
mv "${partial}" "${destination}"
echo "verified draft ranking: ${destination}"
