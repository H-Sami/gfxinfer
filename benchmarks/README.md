# Code-generation benchmark

Run the fixed greedy code corpus with the preferred v9 sequential-W3 artifact:

```bash
./scripts/bench_code.sh
```

An alternate artifact path may be supplied as the first argument. Each run is
stored permanently under `benchmarks/results/<UTC timestamp>/`; no benchmark
input, generated program, or result is placed in a system temporary directory.

The script uses A4 activations, MTP depth 4 for the highly predictable
Fibonacci fixture, and MTP depth 3 for the branchier merge fixture. It preserves
the full CLI output, extracts each fenced Python program, and validates its
syntax with `ast.parse`.
The reported `decode speed` is emitted end-to-end completion throughput. It is
not target-verifier throughput or proposal-only throughput.
