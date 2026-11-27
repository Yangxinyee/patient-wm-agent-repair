The weaker combined edit exceeded the factual-loss limit on tuning stays 100–199, so it is not a safe final candidate. I’ll recover the architecture and cache-building code before screening revisions. A new cache must reproduce the GPU’s factual predictions; otherwise, its local loss estimates cannot guide the edit.

```python
import os, json, inspect, dis, numpy as np, pandas as pd, torch
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD SCHEMA ===")
print(open("/data/README.md").read())
print("=== CURRENT EDITS ===")
print(edited())
print("=== RECOVERY INVENTORY ===")
for k in sorted(list(globals())):
    if k.startswith(("r124", "r129", "r170", "r176", "r183", "c183", "c185", "c196",
                     "head_", "rh_", "eps98", "R85", "train", "tuning", "W0")):
        v = globals()[k]
        if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(k, type(v).__name__, getattr(v, "shape", None),
                  list(v.columns)[:25] if isinstance(v, pd.DataFrame) else "")
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:25])
        elif callable(v):
            try:
                print(k, "function", inspect.signature(v))
            except Exception:
                print(k, "callable")
        else:
            print(k, type(v).__name__, repr(v)[:200])
print("=== ORIGINAL-STATE FORWARD FILES ===")
for d in sorted(os.listdir("/work")):
    if d.startswith("fwd_"):
        fs = [f for f in os.listdir("/work/" + d) if "states" in f or "factual" in f]
        if fs:
            print(d, fs)
print("=== VALUE HEAD TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ("attr_heads.value_bin", "query_norm", "head_context", "z_head",
                            "prior", "register", "numeric")):
        print(n, SHAPES[n])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 198: restart recovery audit\n"
            "Combined fallback is not accepted: factual delta +0.00760 on tuning 100–199. "
            "Recovering architecture and independent factual-cache construction before revising weights. "
            "No weights changed in this audit.\n")
```
