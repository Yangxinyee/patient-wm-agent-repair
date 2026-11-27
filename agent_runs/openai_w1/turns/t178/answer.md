The export succeeded, but it does not show that the 5× dose-embedding edit has corrected the drug effects. I’ll recover the architecture and existing analysis, then compare matched states. A useful dose-sensitive pathway must support the record-derived effects without exceeding the factual-loss limit.

```python
import os, json, inspect, numpy as np, pandas as pd, torch
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== DATA README ===")
print(open("/data/README.md").read())
print("=== CURRENT EDIT ===", edited())
r161_w = np.asarray(load("embed.numeric.weight"))
r161_o = np.asarray(load_original("embed.numeric.weight"))
print("numeric shape:", r161_w.shape)
print("column change norms:", np.linalg.norm(r161_w-r161_o, axis=0).round(5).tolist())
print("=== LIVE FUNCTIONS ===")
print([k for k,v in list(globals().items())
       if inspect.isfunction(v) and not k.startswith("_")])
print("=== LARGE LIVE STRUCTURES ===")
r161_inventory = []
for k,v in list(globals().items()):
    if k.startswith("_") or k.startswith("r161"): continue
    if isinstance(v, pd.DataFrame):
        r161_inventory.append((k, "DataFrame", v.shape, list(v.columns)[:12]))
    elif isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim >= 2 and v.shape[0] > 100:
        r161_inventory.append((k, type(v).__name__, tuple(v.shape)))
print(r161_inventory[:65])
print("=== VOCAB KEYS ===", list(VOCAB))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 161\nForward 11 exported states under the provisional 5x numeric dose/rate edit. Recovering architecture and compact live inventory before matched-state analysis. The edit remains provisional; factual batch delta was +0.00208. Counts: 161 Python runs, 11 forward passes.\n")
```