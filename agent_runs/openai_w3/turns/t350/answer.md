The graft isolates the norepinephrine direction algebraically, but nonlinear rollouts could still change insulin predictions or factual fit. I’ll recover the evaluation IDs and saved checkpoints without changing weights, then test those possibilities. Improved MAP shifts with preserved glucose predictions and factual loss would support keeping it.

```python
from pathlib import Path
import json
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRECORD SCHEMA")
print(Path("/data/README.md").read_text())

tuning_columns299 = list(pd.read_csv("/data/tuning.csv", nrows=0).columns)
stay_columns299 = [c for c in tuning_columns299 if "stay" in c.lower()]
print("\nTUNING COLUMNS:", tuning_columns299)
if stay_columns299:
    stay_column299 = next(
        (c for c in ["stay_id", "stay", "icustay_id"] if c in stay_columns299),
        stay_columns299[0]
    )
    tuning_ids299 = pd.read_csv(
        "/data/tuning.csv", usecols=[stay_column299]
    )[stay_column299].drop_duplicates().tolist()
    print("STAY COLUMN:", stay_column299)
    for j in range(0, len(tuning_ids299), 100):
        print("TUNING COHORT", j // 100,
              json.dumps(tuning_ids299[j:j + 100], default=str))

print("\nCHECKPOINT STRUCTURES")
for key in [
    "checkpoint294", "checkpoint296", "checkpoint287",
    "checkpoint285", "checkpoint292", "checkpoint_alpha01",
    "checkpoint_ne298", "ne_graft_deltas298"
]:
    obj = globals().get(key)
    print(key, type(obj).__name__)
    if isinstance(obj, dict):
        print("keys:", list(obj))
        print("shapes:", {
            k: tuple(v.shape) for k, v in obj.items()
            if hasattr(v, "shape")
        })

print("\nCURRENT EDITS:", edited())
for path in sorted(Path("/work/fwd_36").glob("*.csv")):
    frame299 = pd.read_csv(path)
    print("\nLATEST OUTPUT:", path.name,
          "shape:", frame299.shape, "columns:", list(frame299.columns))
    print(frame299.head(4).to_string(index=False))

notes299 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-IDENTIFICATION NOTES")
print(notes299[:14000])
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 299; forwards 36. Recovered architecture, record schema, "
        "all tuning cohort IDs, checkpoint structures and early identification "
        "notes. No weights changed. NE graft from run298 remains unvalidated; "
        "algebraic directional isolation does not establish rollout isolation. "
        "Next: factual safeguard and paired dose rollouts, then select a "
        "validated checkpoint in the final available Python run.\n"
    )
```
