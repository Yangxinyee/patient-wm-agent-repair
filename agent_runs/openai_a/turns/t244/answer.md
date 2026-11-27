The ridge-1 trial increases factual NLL by 0.00342 nats on these 100 stays, within the 0.005 limit. Its dose effects and broader factual fit remain untested. I’ll recover the architecture, test IDs, and saved edit machinery before checking the rollout. A stronger insulin response with preserved factual fit would support keeping this trial; otherwise I’ll restore the validated fallback.

```python
from pathlib import Path
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== CURRENT EDITS ===")
print(edited())

print("\n=== RELEVANT PERSISTENT VARIABLES ===")
terms = ("checkpoint", "cache", "r3", "gate", "ridge", "contrast",
         "beta", "rho", "tuning", "ids", "trial", "separator")
for name, value in sorted(list(globals().items())):
    if not name.startswith("_") and any(t in name.lower() for t in terms):
        shape = getattr(value, "shape", None)
        detail = ""
        if isinstance(value, dict):
            detail = " keys=" + repr(list(value.keys())[:25])
        elif isinstance(value, (int, float, str)):
            detail = " value=" + repr(value)[:250]
        print(name, type(value).__name__, shape, detail)

tuning_recovery = pd.read_csv("/data/tuning.csv")
print("\n=== TUNING COLUMNS ===")
print(tuning_recovery.columns.tolist())
stay_column_recovery = next(
    c for c in ("stay_id", "stay", "icustay_id") if c in tuning_recovery.columns
)
tuning_ids_recovery = tuning_recovery[stay_column_recovery].drop_duplicates().tolist()
print("\n=== FIRST 100 TUNING IDS ===")
print(tuning_ids_recovery[:100])

notes_recovery = Path("/work/notes.md").read_text()
print("\n=== NOTEBOOK SECTION INDEX ===")
for line in notes_recovery.splitlines():
    if line.startswith("#"):
        print(line)
print("\n=== EARLY CAUSAL-EVIDENCE NOTES ===")
print(notes_recovery[:16000])
```
