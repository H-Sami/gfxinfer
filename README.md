# GFXInfer

> Work in progress. This is an experimental engine for one model on one GPU.
> Expect rough edges, missing features, and changes to the artifact format.

GFXInfer is a native inference engine for the dense Qwen3.8-27B model on the
AMD Radeon RX 9060 XT 16GB (`gfx1200`). It is written in C++20 and HIP and uses
custom low-bit kernels built for this GPU.

The scope is intentionally small. There is no generic graph executor and no
fallback to another inference runtime. The current target is fast code
generation from one request at a time.

## Current status

The engine can:

- convert the official Qwen3.8-27B checkpoint into a checksummed `.gfxi` file;
- load the converted model into one GPU allocation;
- run all 64 text layers, including 48 GDN layers and 16 attention layers;
- use W2, W3, and W4 weights with A4, A8, or FP16 activations;
- run the sequential W3A4 path written for `gfx1200`;
- generate text with the bundled tokenizer and chat template;
- use the model's neural MTP layer for speculative decoding;
- verify up to 31 draft tokens and restore rejected recurrent state;
- capture the MTP proposal chain and full verifier in HIP graphs;
- serve Chat Completions through JSON or SSE streaming.

This is not a finished general-purpose server. Continuous batching, fast
prefill, broad quality testing, tool calls, structured output, authentication,
and long-context validation are still missing.

## Measured speed

These are emitted completion speeds measured on the RX 9060 XT with the
11.644 GiB `code-v9` artifact, A4 activations, and greedy decoding.

| Prompt | MTP depth | Acceptance | Emitted speed |
|---|---:|---:|---:|
| Iterative Fibonacci function | 4 | 84.783% | 72.722 tokens/s |
| Merge two sorted lists | 3 | 85.556% | 62.179 tokens/s |
| Arithmetic mean | | | 67.451 tokens/s |

Both outputs were complete Python programs and passed `ast.parse`. The raw logs
are in `benchmarks/results/20260905T164758Z`.

Some internal paths are faster than the final user-visible number:

| Component | Result |
|---|---:|
| 32-token target verifier | 340.316 evaluated tokens/s |
| Neural MTP proposal | 575.307 proposals/s |
| W3A4 `[17408, 5120]`, batch 5 | 300.102 GB/s |

Those component figures are not decode speed. The current emitted code result
is about 67 tokens/s across the two checked prompts. Near 200 tokens/s remains a
stretch target, but it has not been reached.

The measurements above are development checkpoints, not a broad benchmark
study. See [docs/PERFORMANCE_CONTRACT.md](docs/PERFORMANCE_CONTRACT.md) for the
measurement rules and open performance work.

## Requirements

- AMD Radeon RX 9060 XT 16GB
- ROCm 7.2.3 or newer with `gfx1200` compiler support
- Linux
- CMake 3.24 or newer
- simdjson, ICU, PCRE2, and optionally OpenMP
- enough disk space for the original checkpoint and converted artifact

Other GPUs are rejected on purpose. Supporting a different architecture needs
its own kernels, packing rules, and measurements.

## Build

```bash
git clone https://github.com/H-Sami/gfxinfer.git
cd gfxinfer

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1200

cmake --build build -j
ctest --test-dir build --output-on-failure
./build/gfxinfer probe
```

## Prepare the model

Model weights and converted artifacts are not included in this repository.

Download the pinned source checkpoint and the draft-head ranking:

```bash
./scripts/fetch_qwen38.sh
./scripts/fetch_draft_ranking.sh
./build/gfxinfer verify-source models/Qwen3.8-27B
```

The checkpoint download is large and resumable. Its default speed limit is
33 MiB/s. Set `GFXINFER_DOWNLOAD_LIMIT` to choose another limit.

Convert it into the current code-focused artifact:

```bash
mkdir -p out

./build/gfxinfer convert \
  models/Qwen3.8-27B \
  out/qwen3.8-27b-code-v9-w3seq.gfxi \
  --mlp-bits 3 \
  --head-bits 4 \
  --w3-start-layer 0 \
  --w4-tail 8 \
  --draft-ranking models/ranking.train.counts.i64
```

The reference artifact is 12,502,501,575 bytes with this SHA-256 digest:

```text
0359bdcdcc8a5f03e32ac3b1f4eff1aab19e8ee7c3e1f7f671e36a3671054710
```

The `.gfxi` file is a native, versioned container. It is not GGUF. Details are
in [docs/ARTIFACT_FORMAT.md](docs/ARTIFACT_FORMAT.md).

## Generate from the command line

```bash
./build/gfxinfer generate \
  out/qwen3.8-27b-code-v9-w3seq.gfxi \
  --prompt "Write a Python function that checks whether a string is a palindrome." \
  --max-tokens 256 \
  --context 512 \
  --activation a4 \
  --drafts 4
```

Use `--drafts 0` to disable speculative decoding. Short code prompts generally
work best with the current artifact. Prefill is still serial, so a long prompt
will have noticeably higher time to first token.

## Run the local server

Start the server and wait for the `GFXInfer ready` message:

```bash
./scripts/serve.sh --context 4096 --max-output-tokens 1024
```

Send a streaming request from another terminal:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-gfx1200",
    "messages": [
      {"role": "user", "content": "Write a Python LRU cache. Return only code."}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "stream": true
  }'
```

The server binds to `127.0.0.1` and handles one GPU request at a time. It
supports greedy text Chat Completions, model discovery, health checks, and SSE.
It does not yet support sampling, tools, images, authentication, or TLS. See
[docs/SERVING.md](docs/SERVING.md) for request examples and server options.

## Benchmarks and checks

Run the two code prompts used for the current emitted-speed checkpoint:

```bash
./scripts/bench_code.sh
```

Useful lower-level checks include:

```bash
./build/gfxinfer inspect out/qwen3.8-27b-code-v9-w3seq.gfxi
./build/gfxinfer bench-gemv --format w3 --rows 17408 --cols 5120 --iters 100
./build/gfxinfer spec-bench out/qwen3.8-27b-code-v9-w3seq.gfxi \
  --token 9707 --drafts 4 --tokens 32 --context 128 --activation a4
```

The speculative benchmark compares its output token IDs with serial target
decoding. Performance work is not accepted if that check fails.

## Project layout

```text
include/gfxinfer/    Public C++ interfaces
src/kernels/         HIP kernels for gfx1200
src/                 Artifact, converter, tokenizer, and runtime code
src/server/          Local Chat Completions server
scripts/             Download, benchmark, profiling, and serving helpers
tests/               CPU-side format and quantization tests
docs/                Architecture, format, serving, and roadmap notes
```

## Roadmap

The next major pieces are:

1. Better W3 fusion and fewer activation memory passes.
2. Per-layer differential tests against the official BF16 model.
3. A larger code-generation quality and performance corpus.
4. Faster chunked prefill and prefix caching.
5. Adaptive MTP depth based on accepted tokens per millisecond.
6. Continuous batching and request admission control.
7. Responses API support, tool calls, and structured output.
8. Long-context and stability testing.

The detailed checklist is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Background

The project was inspired by the narrow, hardware-specific approach used by
[NInfer](https://github.com/Neroued/ninfer), but this implementation targets AMD
`gfx1200` and uses its own HIP runtime, kernels, artifact format, and state
management.

## License

The engine source is licensed under Apache-2.0. The Qwen model files are not
part of this repository and remain subject to their own license.
