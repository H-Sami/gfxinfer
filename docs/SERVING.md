# Local serving

`gfxinfer-server` is a separate executable that links directly to the native
GFXInfer engine. It does not launch the CLI as a subprocess and does not change
the validated decode kernels. The model is loaded once at startup; a fresh
transactional decode session is created for each request.

## Start

```bash
cd gfxinfer
./scripts/serve.sh --context 4096 --max-output-tokens 1024
```

Startup takes several seconds while the 11.644 GiB artifact is validated and
uploaded. Readiness is explicit:

```text
GFXInfer ready
  model:    qwen3.8-27b-gfx1200
  endpoint: http://127.0.0.1:8000/v1/chat/completions
```

Use Ctrl-C for a clean shutdown. The default loopback binding is intentionally
not reachable from another machine.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness check |
| `GET` | `/v1/models` | Advertise `qwen3.8-27b-gfx1200` |
| `POST` | `/v1/chat/completions` | Text chat completion, JSON or SSE |

The Chat Completions response and `chat.completion.chunk` envelopes follow the
OpenAI-compatible shapes. With `stream_options.include_usage=true`, the server
emits a final usage chunk before `data: [DONE]`.

This release is deliberately constrained:

- greedy decoding only: omit `temperature` or set it to `0`;
- `n=1` only;
- string message content with `developer`, `system`, `user`, or `assistant` roles;
- no tools, custom stop sequences, structured output, images, authentication,
  or TLS yet;
- one GPU request executes at a time; additional TCP connections wait in the
  listen queue.

Do not bind to a non-loopback address until authentication and TLS are placed
in front of the server.

## curl

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-gfx1200",
    "messages": [
      {"role": "system", "content": "Return concise, correct code."},
      {"role": "user", "content": "Implement binary search in Python."}
    ],
    "max_tokens": 256,
    "temperature": 0,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Remove `"stream": true` and `stream_options` to receive one ordinary JSON
response.

## OpenAI Python client

No OpenAI API key is used by the local server, but the SDK requires a non-empty
placeholder:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

stream = client.chat.completions.create(
    model="qwen3.8-27b-gfx1200",
    messages=[{"role": "user", "content": "Write a Python quicksort."}],
    max_tokens=256,
    temperature=0,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Server options

```text
gfxinfer-server <artifact.gfxi>
  [--host 127.0.0.1]
  [--port 8000]
  [--context 4096]
  [--max-output-tokens 1024]
  [--drafts 0..31]
  [--activation f16|a8|a4|a4w2|a4w4]
```

The convenience launcher selects the preferred v9 artifact and otherwise
passes these options through unchanged.
