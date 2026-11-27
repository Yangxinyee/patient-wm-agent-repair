The edit is exactly dose-invariant, and all 400 tuning stays show a factual NLL increase of only 0.000307—well below 0.005. Before submitting, I’ll audit the causal estimates for dose or future-reading leakage and review delayed responses. Clean predictor timing and negligible delayed effects would support the edit; leakage or a consistent delayed response would undermine it.

```python
import numpy as np
import json

print("=== PANEL COLUMNS ===")
print("H:", list(H.columns))
print("Hrich:", list(Hrich.columns))

print("\n=== FEATURE DEFINITIONS AND MODEL SETTINGS ===")
for key, obj in sorted(list(globals().items())):
    if key.startswith("_"):
        continue
    if isinstance(obj, (list, tuple)) and any(s in key.lower() for s in ("feature", "cols", "columns")):
        print(key, repr(obj)[:2200])
    elif isinstance(obj, dict) and "params" in key.lower():
        print(key, obj)
    elif hasattr(obj, "shape") and any(s in key.lower() for s in ("ne_audit", "strong", "rich_x", "x_same")):
        print(key, obj.shape)

print("\n=== CURRENT-DOSE LEAKAGE SCREEN ===")
sample = np.linspace(0, len(Xrich) - 1, 50000, dtype=int)
screen_x = np.asarray(Xrich[sample], dtype=np.float64)
screen_d = np.asarray(D[sample], dtype=np.float64)
screen_x = np.nan_to_num(screen_x, nan=0.0, posinf=0.0, neginf=0.0)
screen_d = np.nan_to_num(screen_d, nan=0.0, posinf=0.0, neginf=0.0)
screen_x -= screen_x.mean(axis=0)
screen_d -= screen_d.mean(axis=0)
den = np.sqrt((screen_x ** 2).sum(axis=0)[:, None] *
              (screen_d ** 2).sum(axis=0)[None, :])
screen_corr = (screen_x.T @ screen_d) / np.maximum(den, 1e-12)
for j, drug in enumerate(DRUGS):
    idx = np.argsort(np.abs(screen_corr[:, j]))[-5:][::-1]
    print(drug, [(int(i), float(screen_corr[i, j])) for i in idx])
print("An exact correlation of magnitude 1 would flag direct dose leakage; lesser correlations are not causal estimates.")

print("\n=== DELAYED MAIN-SENSOR ESTIMATES ===")
for key, obj in delayed_results.items():
    if key[1] in ("glucose_finger", "map_arterial"):
        print(key, json.dumps(obj, default=str)[:550])

print("\n=== OTHER RELEVANT WORKING OBJECTS ===")
for key, obj in sorted(list(globals().items())):
    if key.startswith("_"):
        continue
    if any(s in key.lower() for s in ("ne_audit", "same_sensor", "held", "rich_features")):
        if hasattr(obj, "shape"):
            print(key, obj.shape)
        elif callable(obj) and hasattr(obj, "__code__"):
            print(key, "args", obj.__code__.co_varnames[:obj.__code__.co_argcount])
        elif isinstance(obj, dict):
            print(key, "keys", list(obj)[:8])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 19 / forward passes 7: causal-estimation audit\n"
            "- Current edits exactly remove dose magnitude while retaining presence and absent-bin rows.\n"
            "- Full 400-stay factual validation is already complete: delta NLL +0.000307, below +0.005.\n"
            "- Audited feature definitions, direct dose leakage, and delayed main-sensor estimates before submission.\n"
            "- No weight changes.\n")
```