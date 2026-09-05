# Architecture

GFXInfer is deliberately closed over one model and one GPU architecture. This
lets the artifact compiler, runtime, and kernels share exact tensor shapes and
physical layouts without a generic operator graph.

## Ownership boundaries

```text
HTTP / CLI gateway
        |
Qwen frontend: tokenizer, fixed chat template, media semantics
        |
Engine: admission, scheduling, lifecycle, publication
        |
Qwen program: model state, KV/GDN transactions, HIP graphs, kernels
        |
gfx1200 device
```

- The gateway owns protocol validation and transport lifetime.
- The frontend owns user-visible Qwen token and template semantics.
- One engine worker owns every mutable request transition.
- The program exclusively owns physical model memory and pending GPU state.
- A token becomes visible only after its KV and GDN state transaction commits.

The execution path does not contain a framework graph, generic tensor registry,
or runtime kernel selection by model shape. Each registered tensor binds to an
expected name, shape, encoding, and kernel consumer during load.

## Request state machine

```text
waiting -> materializing -> prefill -> decode-ready -> terminal-pending -> finished
                                  \-> control-ready -/
```

Cancellation and errors roll back uncommitted KV pages, recurrent GDN state,
MTP draft state, and RNG state before publishing the terminal event.

## Static execution shapes

The current runtime owns one request per decode session and uses fixed GPU
arenas. Its causal verifier executes token widths 1 through 32 and maps one
16-candidate WMMA tile to each wave. Widths above 16 use two waves per output
tile. GDN and attention state advance causally within that block; rejected
speculative suffixes are discarded with compact convolution/key/delta/decay
replay records instead of rerunning all 64 layers.

MTP proposal IDs stay on the device and feed the verifier through indirect
embedding gathers. The complete sequential neural proposal chain and 64-layer
verifier replay as one pre-instantiated HIP graph for the configured speculative
batch; dynamic seed and MTP/verifier positions enter through stable
device-resident scalars. One graph is captured for each recurrent-state parity,
so full-accept pointer swaps remain graph-safe. The round uses one host
synchronization.

After a fully accepted block, only the final MTP K/V entry is advanced; the dead
attention, MLP, proposal-head, and argmax tail is skipped. That cache update uses
the verifier's exact preceding hidden state instead of the draft model's
approximation. Independent-request continuous batching remains a scheduled
milestone. Verifier token width must not be reported as concurrent serving
occupancy until those request-isolation paths exist.

## Model specialization

Qwen3.8-27B contains 64 text layers:

- 48 gated delta-network recurrent layers;
- 16 grouped-query full-attention layers at every fourth position;
- hidden width 5120 and MLP width 17408;
- 24 query heads, four KV heads, and head width 256;
- one full-attention MTP layer.

The target package defines the complete ordered tensor inventory and fusion row
order. An unknown model revision, tensor, shape, layout, or numeric recipe is a
load error.

## Clean-room boundary

The implementation is written from public model mathematics, AMD/LLVM
documentation, and independently specified behavior. No code or build artifact
from another local inference project is linked, copied, or treated as a runtime
dependency.
