# Execution roadmap

Every milestone ends in an executable test or benchmark. Later stages may tune
an established contract but may not replace a failed path with another inference
runtime.

## M0 - executable foundation

- [x] New standalone C++20/HIP build targeting only `gfx1200`.
- [x] Strict device discovery and rejection of other GPU architectures.
- [x] `.gfxi` v1 mapping, atomic writing, bounds checks, and checksums.
- [x] Deterministic W2/W3/W4 group-128 packers.
- [x] Baseline native low-bit GEMV kernels with validation and timing.
- [x] Tune tiled W2/W4 A4/A8/FP16 GFX12 WMMA through two-wave B32 execution.
- [x] Add versioned sequential-W3 packing and a gfx1200 bulk bit-dilation path.
- [x] Register the exact 1,199-name and shape official Qwen source inventory.
- [x] Stream-safe safetensors shard/index parser with bounds and dtype checks.
- [x] Atomic random-access `.gfxi` streamer with incremental checksums.
- [x] Emit and checksum-validate a complete 9.659GiB text+MTP bootstrap artifact.
- [x] Single-allocation aligned GPU weight arena and asynchronous resident upload.
- [x] Add permanent code-generation fixtures with emitted-speed and syntax checks.
- [x] Add a durable GPU power/busy/clock profiler harness.
- [ ] Add a machine-readable benchmark result format.

## M1 - exact Qwen text artifact

- [x] Add exact shapes and bind the registered source inventory to a fixed revision.
- [x] Bind exact tensor shapes and source revisions while streaming shards.
- Implement activation capture, channel scaling, GPTQ error correction, outlier
  selection, and global bit allocation.
- Emit `maxspeed-w2w4-a4a8-v1` and `balanced-w4-a8-v1` artifacts plus reports.
- Enforce the 9.25GiB resident weight/MTP limit and the quality gates.
- [x] Implement the tokenizer and fixed Qwen chat-template behavior.

## M2 - Qwen program

- [x] BF16/FP32 normalization, rotary, residual, and greedy argmax kernels.
- [x] GQA decode attention, Q/K head norms, rotary, and contiguous KV cache.
- [x] MLP gate/up/SiLU/down execution with activation-quantization reuse.
- [x] Exact 48-layer GDN recurrence and four-tap convolution.
- [x] Full target vocabulary projection and a 98,304-row active ranked MTP proposal shortlist.
- Per-operation and per-layer differential tests against official BF16 math.

## M3 - fast text release

- [x] Capture the complete MTP proposal and 64-layer verifier round, including
  position-dependent attention, for the configured batch and both state parities.
- [ ] Add HIP virtual-memory KV pages.
- [x] Transactional KV/GDN state with compact accepted-prefix restoration.
- [x] Built-in device-resident MTP drafting and exact verification at depths 1..31.
- Online MTP depth selection based on accepted tokens per millisecond.
- Prefix cache with device and pinned-host checkpoint tiers.
- Bounded continuous batching and memory admission control.
- [x] Add single-request OpenAI Chat Completions and incremental SSE serving.
- [ ] Add the Responses API, tool calls, structured output, authentication, and TLS.
- Meet correctness, quality, memory, stability, and performance release gates.

## M4 - complete service parity

- Anthropic Messages protocol and tool-call formatting.
- JSON-schema DFA compilation and constrained draft/target sampling.
- Vision transformer and merger kernels.
- Image and video acquisition, validation, preprocessing, and prompt caching.
- Vision-enabled startup profile that does not reduce text-only residency.
- Long-context and concurrency campaigns through the 262,144-token limit.
