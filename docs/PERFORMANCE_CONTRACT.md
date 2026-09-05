# Performance contract

Performance claims must be reproducible and paired with correctness results.

## Fixed target

- Radeon RX 9060 XT 16GB (`gfx1200`)
- stable GPU power state and recorded temperature/clocks
- one resident Qwen3.8-27B artifact
- ROCm and compiler revisions recorded verbatim

## Release gates

| Workload | Required | Goal |
|---|---:|---:|
| Single-request MTP0 decode | 27 token/s | 32-36 token/s |
| Mixed single-request adaptive MTP | 55 token/s | 70-90 token/s |
| Structured/code adaptive MTP | 100 token/s | 130-160 token/s |
| Four-request aggregate decode | 130 token/s | 170-220 token/s |
| 4K prefill | 1,200 token/s | 2,000+ token/s |

Near 200 token/s is a stretch single-request result limited to prompts where MTP
acceptance is high. It is never substituted for general decode throughput.

## Current measured snapshot

Measured on the target RX 9060 XT with the tiled 11.644 GiB code-v9 artifact on
2026-09-05. These are single-run development checkpoints; the committed corpus
runner preserves future repeated-run logs under `benchmarks/results/`.

| Path | Result | Meaning |
|---|---:|---|
| Fibonacci, A4 + MTP4 | 72.722 token/s | 101 emitted timed tokens; 84.783% acceptance; valid complete Python |
| Merge sorted, A4 + MTP3 | 62.179 token/s | 107 emitted timed tokens; 85.556% acceptance; valid complete Python |
| Two-fixture arithmetic mean | 67.451 token/s | emitted single-request code decode, not verifier throughput |
| Causal verifier B5 | 95.422 token/s | evaluated target tokens, not emitted tokens; 16-block v9 run |
| Causal verifier B32 | 340.316 token/s | evaluated target tokens, not emitted or served tokens; 8-block v9 run |
| MTP proposal step | 575.307 steps/s | v9 96K-head measurement; proposal-only speed; target verification excluded |
| W3A4 `[17408,5120]`, B5 | 300.102 GB/s | v9 sequential-W3 microkernel bandwidth; exact sampled validation |

The 72.722 token/s run exceeds the earlier 67-68 token/s code checkpoint. The
complete proposal-plus-verifier graph retained exact serial token parity; its
gain over verifier-only graph replay was noise-scale, identifying kernel and
memory work rather than host launch structure as the next bottleneck. The wider
gap on `merge_sorted` is explained by MTP acceptance, not by substituting a
slower execution backend. Near-200 emitted token/s remains the stretch
objective, but verifier and proposal-only rates are never reported as
user-visible decode.

## Measurement rules

- Warm the persistent process and every measured HIP graph first.
- Exclude model loading and client tokenization from kernel throughput.
- Include scheduling and server execution in serving throughput.
- Disable prefix reuse for baseline results and report it separately.
- Record prompt tokens, completion tokens, batch occupancy, KV format, MTP depth,
  drafted/accepted tokens, speculative rounds, and tokens per round.
- Audit response structure and finish reason separately from speed.
- Report arithmetic mean, standard deviation, median, and p10/p90 across fixed
  prompts and seeds.
- Preserve raw per-request JSON and profiler outputs.
- Preserve power, GPU-busy, memory-busy, and clock telemetry under the project
  benchmark directory; do not use transient paths.

## Optimization gates

The packed decode GEMV must sustain at least 75% of practical device bandwidth
for its shape before model-level fusion work is accepted. Prefill and MTP
verification kernels must separately report achieved INT4 operations, LDS use,
VGPR pressure, occupancy, and memory traffic through ROCprofiler.
