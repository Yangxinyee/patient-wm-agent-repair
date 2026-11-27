The gated decoder looks promising, but it does not yet establish that the weight edit preserves factual fit or improves rollouts. I’ll recover the record-based effect estimates and validation setup first. The edit should survive only if factual NLL stays within the 0.005 budget and insulin rollouts move toward those estimates.

```python
import os, json, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== EARLY NOTEBOOK: RECORD-BASED EVIDENCE ===")
notes_text = open("/work/notes.md").read()
print(notes_text[:12000])

print("\n=== PERSISTENT VARIABLE INDEX ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    obj = globals()[name]
    if isinstance(obj, np.ndarray):
        print(name, "ndarray", obj.shape, obj.dtype)
    elif hasattr(obj, "shape") and not isinstance(obj, type):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict keys:", list(obj.keys())[:18])
    elif isinstance(obj, (list, tuple)) and len(obj) <= 12:
        print(name, repr(obj)[:250])

print("\n=== FORWARD OUTPUT FILES ===")
for k in range(1, 7):
    folder = f"/work/fwd_{k}"
    if os.path.isdir(folder):
        print(folder, os.listdir(folder))

print("\n=== TUNING RECORD HEADER ===")
tuning_header = pd.read_csv("/data/tuning.csv", nrows=5)
print(tuning_header.to_string(index=False))
print("\nCurrent edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 54: restart recovery\n"
            "The provisional gated insulin head edit remains installed but unvalidated. "
            "Recovering architecture, causal evidence, and validation identifiers before "
            "paired factual and dose-rollout tests. No additional weights changed.\n")
```