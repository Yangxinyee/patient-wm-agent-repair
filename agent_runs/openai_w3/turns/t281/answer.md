The reconstruction check shows that candidate250 changes the actual attention payload, not just a downstream approximation. Its factual fit and rollout effects are still untested. I’ll recover the validation stays and architecture first; improvement requires both the 0.005-nat factual limit and dose responses closer to the record-derived effects.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

notes251 = Path("/work/notes.md").read_text()
print("=== EARLY IDENTIFICATION NOTES ===")
print(notes251[:7000])

facts251 = {}
for model251 in ("original", "edited"):
    path251 = Path(f"/work/fwd_15/factual_{model251}.csv")
    if path251.exists():
        facts251[model251] = pd.read_csv(path251)
        print("fwd_15", model251, facts251[model251].shape,
              facts251[model251].columns.tolist())
        print(facts251[model251].head(2).to_string(index=False))

if facts251:
    frame251 = next(iter(facts251.values()))
    stay_col251 = next(
        (c for c in ("stay_id", "stay", "icu_stay_id") if c in frame251.columns),
        None
    )
    if stay_col251 is not None:
        stays_eval251 = frame251[stay_col251].drop_duplicates().tolist()[:100]
        stays_eval251 = [
            int(x) if isinstance(x, (int, np.integer, float, np.floating))
            else str(x) for x in stays_eval251
        ]
        print("VALIDATION STAYS:", stays_eval251)

print("=== RELEVANT RETAINED VARIABLES ===")
for name251 in sorted(list(globals())):
    if any(s in name251.lower() for s in
           ("checkpoint249", "checkpoint193", "candidate250",
            "gain250", "dd193", "carrier250")):
        value251 = globals()[name251]
        print(name251, type(value251).__name__,
              getattr(value251, "shape", ""),
              list(value251)[:15] if isinstance(value251, dict) else "")

with Path("/work/notes.md").open("a") as log251:
    log251.write(
        "\nPython run 251; forwards 18. Restart recovery: candidate250 remains "
        "installed and unvalidated. Read architecture/data documentation and "
        "recovered prior factual-validation stay IDs before authoritative tests. "
        "No weights changed.\n"
    )
```