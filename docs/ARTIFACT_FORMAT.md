# `.gfxi` artifact format, version 1

The runtime consumes one immutable, little-endian, memory-mappable artifact.
GGUF is not an execution format. Conversion performs all validation, fusion,
packing, swizzling, and quantization before the server starts.

## File layout

```text
offset 0     256-byte header
offset 256   tensor directory (128 bytes per entry)
             concatenated tensor-name bytes
             padding to 256 bytes
             individually aligned tensor payloads and auxiliary planes
```

All offsets are absolute. Every payload begins at a power-of-two alignment of at
least 256 bytes. Regions may not overlap.

## Identity

The header registers four independent strings:

- `target_arch`: exactly `gfx1200`;
- `model_id`: exactly `qwen3.8-27b`;
- `weights_id`: the calibrated weight profile;
- `recipe_id`: the byte-exact conversion recipe revision.

Filenames never select behavior. The runtime fails closed on an unknown
identity.

## Integrity

- The header records the exact file size and section bounds.
- CRC64-ECMA covers the directory and string table.
- CRC64-ECMA covers every primary tensor payload.
- Tensor names are unique, relative, and restricted to portable path bytes.
- The production converter additionally writes a conversion report containing
  source SHA-256 identities and will add whole-artifact SHA-256 in format v2.

## Numeric formats

- `W2G128`: four signed two-bit values per byte. Codes map to `{-3,-1,1,3}`.
- `W3G128`: 3-bit codes map to `[-4,3]`. Legacy tiled recipes store three
  16-byte bitplanes per 128-value group. The `w3seq` v9 recipes instead store
  four 12-byte sequential 3-bit fragments per group; the gfx1200 kernel expands
  eight codes at once with a staged bit-dilation network. Recipe identity, not
  the filename, selects the decode path.
- `W4G128`: two signed two's-complement nibbles per byte.
- `W8G128`: signed bytes with group scales.
- `FP8_ROW`: FP8 values with one registered row scale.

Groupwise formats quantize across input columns. Scale and optional outlier
planes live in the auxiliary region and their exact internal schema is selected
by `recipe_id`; the target binder does not infer missing fields.

Version 1 establishes framing and low-bit layouts. Adding the complete Qwen
inventory will not weaken any existing validation rule.
