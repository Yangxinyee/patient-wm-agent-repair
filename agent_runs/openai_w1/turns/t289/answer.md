The quarter-strength edit did not strengthen the insulin response or produce the expected positive norepinephrine response. I will recover the architecture, factual-fit result, and existing causal estimates before choosing another weight path. A useful edit must improve dose responses without exceeding the factual-loss limit.

```python
import os, json, numpy as np, pandas as pd, torch, inspect

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== FORWARD 26 FILES ===")
print(os.listdir("/work/fwd_26"))
for fn in os.listdir("/work/fwd_26"):
    if fn.endswith(".json"):
        print(fn, open("/work/fwd_26/" + fn).read()[:6000])

print("\n=== RECENT KERNEL OBJECTS ===")
_items253 = list(globals().items())
for k, v in _items253:
    if k.startswith("_"):
        continue
    if any(s in k.lower() for s in
           ["236", "239", "245", "246", "247", "249", "251", "252",
            "target", "causal", "dynamic", "carrier", "relay", "effect"]):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape, list(v.columns)[:16])
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:20])
        elif callable(v):
            print(k, "callable", inspect.signature(v))
        else:
            print(k, type(v).__name__, str(v)[:450])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 253\n"
            "Forward27 quarter-strength selective marker did not improve the desired rollout effects "
            "(insulin x2 -2.003, half .818; NE x2 -.074, half -.154). "
            "Recovering architecture, factual26, and existing identification objects before choosing "
            "a different weight route. No weights changed. Counts: 253 Python runs, 27 forwards.\n")
```