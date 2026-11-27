The last calculation stopped on a feature-width mismatch; it did not produce a candidate edit. The state reconstruction was close but not exact. I’ll recover the architecture, causal estimates, and cached variables before fixing that mismatch. A consistent reconstruction would support proceeding; unexplained discrepancies would argue against editing this pathway.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())

notes = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND TARGETS ===")
print(notes[:16000])

print("\n=== SAVED CAUSAL ESTIMATES ===")
import pandas as pd
for filename in ["dml_results.csv", "dynamic_iv.csv", "sensor_iv.csv",
                 "kernel_targets.csv", "local_projections.csv"]:
    p = Path("/work") / filename
    if p.exists():
        t = pd.read_csv(p)
        print(filename, t.shape, list(t.columns))
        print(t.head(12).to_string(index=False))

print("\n=== KERNEL INVENTORY ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB", "notes", "t"}:
        continue
    if isinstance(obj, np.ndarray):
        print(name, "array", obj.shape, obj.dtype)
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns[:18]))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), list(obj.keys())[:12])
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(name, "function", getattr(getattr(obj, "__code__", None), "co_varnames", ())[:18])

print("\nACTIVE EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 73: restart recovery\n"
            "- Run 72 stopped at a 17-versus-16 feature-width mismatch; no candidate edit was completed.\n"
            "- Recovering architecture, identification results, and cached state-reconstruction variables.\n"
            "- Forward passes used: 11. No active edits.\n")
```