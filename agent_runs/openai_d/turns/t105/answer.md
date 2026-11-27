The saved probe and diagnostic variables are still available, but the inventory again exceeded the output limit. No weights changed. I’ll print only the causal estimates and probe inputs now. The probe must match across reading types and register settings before I use it to construct a correction.

```python
import dis
from pathlib import Path
import numpy as np

print("=== CAUSAL NOTES ===")
print(Path("/work/notes.md").read_text()[:4800])

print("\n=== PROBE DIAGNOSTICS ===")
for name in [
    "r90_result", "r90_results", "r90_spec", "r90_coef", "r90_numeric",
    "r90_statusW", "r90_side", "r91_results", "r91_best",
    "r91_gaps", "r91_scale", "r91_tm", "r91_tn",
    "rh_rcfg", "rh_cfg", "rh_chosen"
]:
    if name not in globals():
        continue
    obj = globals()[name]
    if hasattr(obj, "shape"):
        print(name, "shape", tuple(obj.shape))
        if np.prod(obj.shape) <= 50:
            print(np.asarray(obj))
    else:
        print(name, repr(obj)[:450])

print("\n=== PROBE TABLES ===")
for name in ["r90_meta", "r87_probe", "rh_meta"]:
    obj = globals().get(name)
    if hasattr(obj, "columns"):
        print(name, "columns:", list(obj.columns))
        print(obj.head(2).to_string(index=False))

print("\n=== LAST-NUMERIC WEIGHTS ===")
for name in NAMES:
    if name.startswith("last_"):
        print(name, SHAPES[name])

print("\n=== FAILED HELPER ===")
dis.dis(r91_mlp)

print("\nCURRENT EDITS:", edited())
```
