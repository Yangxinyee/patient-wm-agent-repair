```python
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

# Preserve the current candidate without changing any submitted weights.
backup297_current = {
    n: torch.as_tensor(load(n)).detach().clone() for n in edited()
}
print("CURRENT EDITS:", list(backup297_current))
for objname in ["backup296", "grid286"]:
    obj = globals().get(objname)
    print(objname, type(obj).__name__)
    if isinstance(obj, dict):
        print("keys:", list(obj.keys())[:40])
        if objname == "backup296":
            for n in backup297_current:
                if n in obj:
                    a = torch.as_tensor(obj[n])
                    b = backup297_current[n]
                    print(n, tuple(a.shape),
                          "max current-minus-fallback:",
                          float((a-b).abs().max()))

audit297_df = pd.read_csv("/data/tuning.csv")
print("TUNING COLUMNS:", audit297_df.columns.tolist())
audit297_staycol = next(
    c for c in audit297_df.columns
    if c.lower() in ("stay_id", "stay", "icustay_id")
)
audit297_ids = audit297_df[audit297_staycol].drop_duplicates().tolist()
for j in range(4):
    print(f"AUDIT COHORT {j}:",
          audit297_ids[100*j:100*(j+1)])

print("=== AVAILABLE FIT / TARGET OBJECT NAMES ===")
print([n for n in list(globals())
       if re.search(r"(kernel|target|causal|dml|weighted281|grid286|audit.*ids)", n, re.I)
       and not n.startswith("_")])

notes297 = Path("/work/notes.md").read_text()
sections297 = re.split(r"(?m)(?=^## )", notes297)
evidence297 = [
    s for s in sections297
    if re.search(r"(corrected DML|threshold|run 23\b|quiet.channel|causal.identification)",
                 s, re.I)
]
print("=== RELEVANT NOTE SECTIONS ===")
print([s.splitlines()[0] for s in evidence297][-20:])
for s in evidence297[-5:]:
    print(s[:2400])

print("=== LATEST FACTUAL CSV SCHEMA ===")
latest297 = pd.read_csv("/work/fwd_37/factual_edited.csv")
print(latest297.columns.tolist())
print(latest297.head(2).to_string(index=False))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 297 / final-selection recovery\n"
        "No weights changed. Preserved current NE-focused candidate in "
        "backup297_current; recovered architecture, schema, fallback structure, "
        "causal-note pointers, and all four tuning audit cohorts. "
        "Latest cohort factual delta is +.00515, so current candidate needs "
        "broader auditing and possibly interpolation toward backup296.\n"
    )
```
