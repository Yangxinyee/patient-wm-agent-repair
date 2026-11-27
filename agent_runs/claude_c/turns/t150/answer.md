My last attempt's output was blocked, so I'll keep replies short and outputs small. Run 116 showed the prelude attention has separate `q` (192×192) and `kv` (384×192) projections. `encode_action` has its own two MLPs, size term and output norm. Any edit to `encode_action.*` touches only the action token, so it cannot disturb how readings are encoded. That makes it the safest selective lever, but its output is layer-normalised, which is the likely cause of saturation.

Next I'll record this in the notebook. I'll also confirm the op list is empty before designing an `encode_action`-side edit. That edit shrinks the constant (dose-independent) part of the action token relative to the dose-dependent part. The hypothesis is confirmed if the insulin and NE responses grow beyond the input-scaling saturation (about −2.5) with factual NLL unchanged.

```python
with open('/work/notes.md','a') as f:
    f.write("\n## Restart-7 (runs 115-117)\n- Cleared stale prelude.attn.out op (ops empty).\n"
            "- Prelude attn: q (192x192), kv (384x192), out. encode_action: mlp1 (768), mlp2 (768), size, norm (only touches action token -> selective).\n"
            "- Keep outputs short (classifier blocked long output).\n")
print(ops(), len(open('/work/notes.md').read()))
```