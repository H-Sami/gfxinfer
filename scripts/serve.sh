#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact="${GFXINFER_ARTIFACT:-${project_root}/out/qwen3.8-27b-code-v9-w3seq.gfxi}"

if [[ ! -x "${project_root}/build/gfxinfer-server" ]]; then
  echo "error: build/gfxinfer-server is missing; build the project first" >&2
  exit 1
fi
if [[ ! -r "${artifact}" ]]; then
  echo "error: cannot read artifact: ${artifact}" >&2
  exit 1
fi

exec "${project_root}/build/gfxinfer-server" "${artifact}" "$@"
