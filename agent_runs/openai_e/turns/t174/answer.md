The new correction looks mild in the local diagnostic, but full-sequence factual fit is still untested. I’ll recover the architecture, earlier evidence, and validation stay IDs before requesting that test. A factual NLL increase above 0.005 would reject this candidate.

```python
from pathlib import Path
import re
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())

notes152 = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD EVIDENCE ===")
print(notes152[:9000])

sections152 = re.split(r"(?=^## )", notes152, flags=re.M)
print("\n=== PREVIOUS FEASIBLE-EDIT EVIDENCE ===")
selected152 = [
    s for s in sections152
    if (("baseline77" in s or ".35" in s or "0.35" in s)
        and ("NLL" in s or "readout" in s or "rollout" in s))
]
for s in selected152[-7:]:
    print(s[:2200])

print("\n=== MOST RECENT VALIDATION STAYS ===")
factual152 = pd.read_csv("/work/fwd_16/factual_original.csv")
print("Columns:", factual152.columns.tolist())
print(factual152.head(3).to_string(index=False))
staycols152 = [c for c in factual152.columns if "stay" in c.lower()]
if staycols152:
    validation_stays152 = factual152[staycols152[0]].drop_duplicates().tolist()
    print("Validation stay IDs:", validation_stays152)

print("\n=== LIVE FIT OBJECTS ===")
for nm152 in ["Bdir141", "Binput142", "fits150", "baseline77", "fits144"]:
    obj152 = globals().get(nm152)
    print(nm152, type(obj152).__name__, getattr(obj152, "shape", None),
          list(obj152.keys()) if isinstance(obj152, dict) else "")
print("Current edits:", edited())

with open("/work/notes.md", "a") as f152:
    f152.write("\n## Run 152: restart recovery before GPU validation\n"
               "Recovered architecture, record definitions, earlier effect evidence, and factual-validation stay IDs. "
               "Candidate151 remains unaccepted; no weights changed. Python runs: 152; forwards: 16.\n")
```