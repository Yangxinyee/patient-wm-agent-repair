The architecture confirms that doses enter the recurrent state through the action encoder and latent posterior. The notebook’s record-based estimates indicate effects much larger than the original rollouts, but the calibration details were cut off. I will inspect the retained helpers and candidate tensors; correctly aligned latent and state calculations would let us test the pathway without repeating the earlier reconstruction work.

```python
import inspect, numpy as np, torch
print("=== RECENT OBJECT INVENTORY ===")
items173 = []
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if any(k.endswith(str(j)) for j in range(153, 172)):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            s = f"{k}: {type(v).__name__} {tuple(v.shape)} {v.dtype}"
        elif isinstance(v, dict):
            s = f"{k}: dict {list(v.keys())[:12]}"
        elif callable(v):
            s = f"{k}: function {inspect.signature(v)}"
        elif isinstance(v, (list, tuple)):
            s = f"{k}: {type(v).__name__} len={len(v)}"
        elif isinstance(v, (str, int, float, bool)):
            s = f"{k}: {repr(v)[:100]}"
        else:
            continue
        items173.append(s)
print("\n".join(items173)[:6500])
print("\n=== REPLAY / READOUT FUNCTION NAMES ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if callable(v) and any(s in k.lower() for s in
        ["core", "prelude", "read", "post", "action", "embed", "replay", "view", "payload"]):
        try:
            print(k, inspect.signature(v))
        except Exception:
            pass
print("\n=== CHECKPOINT ===")
print("Edited:", edited())
print("Zero checkpoint:", [(k, tuple(v.shape)) for k,v in zero169.items()] if "zero169" in globals() else "missing")
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 173; forwards 10. Architecture and early record-only estimates recovered. Run 172 output truncated before retained-object inventory; requesting compact recent inventory. Existing zero-payload transport edits unchanged.\n")
```