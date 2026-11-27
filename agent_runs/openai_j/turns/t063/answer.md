The reconstruction is exact for drug-only bundles, but not for observations or END bundles. That rules out an atom-count error; it does not yet justify a weight edit. I’ll recover the architecture, causal evidence, and cached helpers before choosing the next test. A useful repair must improve dose responses while passing the factual-NLL check.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK: IDENTIFICATION EVIDENCE ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:11000])

print("\n=== AVAILABLE KERNEL OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    obj = globals()[name]
    if isinstance(obj, (pd.DataFrame, np.ndarray)):
        print(name, type(obj).__name__, obj.shape)
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        code = getattr(obj, "__code__", None)
        print(name, "function", code.co_varnames[:code.co_argcount] if code else "")
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj)[:8])

print("\n=== SAVED ANALYSIS TABLES ===")
for filename in ["dynamic_iv.csv", "proxy_iv.csv", "sensor_iv.csv",
                 "local_projections.csv", "dml_results.csv", "kernel_targets.csv"]:
    frame = pd.read_csv("/work/" + filename)
    print(filename, frame.shape, frame.columns.tolist())
    print(frame.head(3).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 53: restart recovery\n"
            "- Drug-only posterior reconstruction is exact; observation/END reconstruction remains unvalidated.\n"
            "- Recovering architecture, identification evidence, and cached helpers before another reversible edit.\n"
            "- No active edits; nine forward passes used.\n")
```
