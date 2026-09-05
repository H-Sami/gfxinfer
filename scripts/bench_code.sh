#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact="${1:-${project_root}/out/qwen3.8-27b-code-v9-w3seq.gfxi}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${project_root}/benchmarks/results/${stamp}"
mkdir -p "${result_dir}"

run_case() {
  local name="$1"
  local maximum_tokens="$2"
  local draft_depth="$3"
  local prompt="$4"
  local log="${result_dir}/${name}.log"
  local code="${result_dir}/${name}.py"

  "${project_root}/build/gfxinfer" generate "${artifact}" \
    --prompt "${prompt}" --max-tokens "${maximum_tokens}" \
    --context 512 --activation a4 --drafts "${draft_depth}" | tee "${log}"
  awk '/^```python$/ {inside=1; next} /^```/ {inside=0; next} inside' \
    "${log}" > "${code}"
  python3 -c 'import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())' \
    "${code}"
}

run_case fibonacci 120 4 \
  'Write a Python function fibonacci(n: int) -> int that returns the nth Fibonacci number. Use an iterative implementation, raise ValueError for negative n, and return only a fenced Python code block.'

run_case merge_sorted 160 3 \
  'Return only a fenced Python code block defining merge_sorted(left: list[int], right: list[int]) -> list[int]. It must merge two already-sorted lists in linear time without calling sorted().'

echo "code benchmark logs and syntax-checked outputs: ${result_dir}"
