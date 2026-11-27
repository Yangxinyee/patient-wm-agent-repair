The architecture confirms that doses enter through a separate action encoder, while value predictions also use a last-reading shortcut. The inventory was truncated, so I still need the retained state and causal-analysis objects. I’ll retrieve a compact listing; those objects should let me test a correction without repeating the reconstruction.

```python
import inspect, numpy as np, pandas as pd, torch
_s69 = list(globals().items())
print("DATAFRAMES:", "; ".join(
    f"{n}:{v.shape}" for n, v in _s69
    if not n.startswith("_") and isinstance(v, pd.DataFrame)))
print("ARRAYS:", "; ".join(
    f"{n}:{tuple(v.shape)}" for n, v in _s69
    if not n.startswith("_") and isinstance(v, (np.ndarray, torch.Tensor))
    and np.prod(v.shape) >= 500))
print("DICTS:")
for n, v in _s69:
    if not n.startswith("_") and isinstance(v, dict) and n not in ("VOCAB", "SHAPES"):
        print(n, list(v.keys())[:12])
print("FILES:", {
    d: os.listdir("/work/" + d)
    for d in os.listdir("/work") if d.startswith("fwd_")
})
print("VALUE/CORE TENSORS:", {
    n: SHAPES[n] for n in NAMES
    if any(s in n for s in ("attr_heads.value", "core.c_out", "core.b_in",
                             "core.z_", "last_numeric", "encode_action"))
})
print("SOURCE ACCESS:")
for n in ("memory67", "self_block67", "head_features_fixed",
          "build_history_features", "persistence_loss"):
    try:
        source = inspect.getsource(globals()[n])
        print(n, "source length", len(source), "\n", source[:1300])
    except Exception as e:
        print(n, type(e).__name__, str(e)[:100])
```
