The stronger head passes this factual audit, but its +0.00452-nat increase leaves little margin. Its rollout benefit is still untested. I’ll recover the fitting machinery and records-based evidence before deciding whether to retain it; better dose responses without exceeding the factual limit would support it.

```python
from pathlib import Path
import inspect, json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

notes249 = Path("/work/notes.md").read_text()
print("=== EARLY EVIDENCE / DECISIONS ===")
print(notes249[:16000])

print("=== LIVE FUNCTIONS ===")
print(sorted(k for k, v in list(globals().items())
             if inspect.isfunction(v) and not k.startswith("_")))
print("=== RELEVANT LIVE OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if not k.startswith("_") and any(s in k.lower() for s in
        ["235", "237", "242", "248", "kernel", "target", "audit", "tuning",
         "dose", "causal", "feature", "backup"]):
        print(k, type(v).__name__, getattr(v, "shape", ""),
              list(v.keys())[:12] if isinstance(v, dict) else "")

f249 = pd.read_csv("/work/fwd_20/factual_original.csv")
print("=== LATEST AUDIT COLUMNS ===", f249.columns.tolist())
print(f249.head(3).to_string(index=False))
staycol249 = next(c for c in f249.columns if "stay" in c.lower())
ids249 = f249[staycol249].drop_duplicates().tolist()
print("=== PAIRED ROLLOUT REQUEST ===")
print(json.dumps({"type": "dose_edit", "model": "both",
                  "split": "tuning", "stays": ids249}))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 249 / stronger-head audit recovery\n"
            "Forward19 factual: original 1.61582, edited 1.62034, "
            "delta +0.00452 on 4664 readings / 100 independent tuning stays. "
            "Within tolerance but close to limit. Current grid242[(.5,.4)] "
            "causal rollout remains unvalidated; recovering evidence and "
            "live interfaces before further changes. No weights changed.\n")
```