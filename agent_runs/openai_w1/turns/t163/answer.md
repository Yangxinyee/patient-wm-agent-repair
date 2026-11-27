The architecture confirms that doses enter the posterior and recurrent state, not just the value head. That gives us possible intervention pathways, but none is validated yet. I’ll inspect the cached head inputs and action encodings next; a useful pathway must retain dose information that can be separated from the treatment policy.

```python
import inspect, numpy as np, torch
print("CONFIG:", VOCAB["config"])
print("\nARRAY INVENTORY:")
items = []
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        items.append(f"{name}:{tuple(obj.shape)}")
print(" | ".join(sorted(items)))
print("\nDICTIONARY INVENTORY:")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"SHAPES", "VOCAB", "W", "globals"}:
        continue
    if isinstance(obj, dict):
        print(name, len(obj), str(list(obj)[:5])[:170])
print("\nOTHER REPLAY/HEAD OBJECTS:")
for name, obj in sorted(list(globals().items())):
    if any(s in name.lower() for s in ("head", "cache", "replay", "rx_", "repair_")):
        if not isinstance(obj, (dict, np.ndarray, torch.Tensor)) and not inspect.isfunction(obj):
            print(name, type(obj).__name__, str(obj)[:150])
print("\nRELEVANT TENSORS:")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_in", "z_head", "out_norm", "encode_action", "embed.numeric")):
        print(name, SHAPES[name])
print("\nHEAD CONSTRUCTION:")
rx_show_function("ce_make_heads", limit=3300)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 149\nArchitecture confirms action embeddings affect posterior and recurrent state; no validated localized pathway yet. Recovering compact cache inventory and head construction to avoid truncated diagnostics. No edits. Counts: 149 Python runs, 8 forward passes.\n")
```