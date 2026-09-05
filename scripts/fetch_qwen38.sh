#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
destination="${1:-models/Qwen3.8-27B}"
base="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/${revision}"
rate_limit="${GFXINFER_DOWNLOAD_LIMIT:-33M}"

mkdir -p "${destination}"

resource_files=(
  LICENSE
  chat_template.jinja
  config.json
  crc32.txt
  generation_config.json
  merges.txt
  model.safetensors.index.json
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  video_preprocessor_config.json
  vocab.json
)

download() {
  local filename="$1"
  echo "fetching ${filename}"
  curl --location --fail --show-error --silent \
    --retry 8 --retry-all-errors --continue-at - --limit-rate "${rate_limit}" \
    --output "${destination}/${filename}" \
    "${base}/${filename}?download=true"
}

for filename in "${resource_files[@]}"; do
  download "${filename}"
done

for shard in {1..18}; do
  printf -v filename 'model-%05d-of-00018.safetensors' "${shard}"
  download "${filename}"
done

echo "download complete at ${destination}"
echo "pinned revision: ${revision}"
