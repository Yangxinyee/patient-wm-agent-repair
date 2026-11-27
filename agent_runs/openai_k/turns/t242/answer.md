The edit preserved factual fit: the loss increased by only 0.00002 nats per reading. That warrants a rollout test, but does not establish correct drug effects. I’ll recover the architecture and exact stay IDs first; rollout shifts closer to the record-derived targets would support keeping this candidate.

```python
from pathlib import Path
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

resume219_tuning = pd.read_csv("/data/tuning.csv")
print("Tuning columns:", resume219_tuning.columns.tolist())
resume219_idcol = next(
    (c for c in ["stay_id", "stay", "id"] if c in resume219_tuning.columns),
    resume219_tuning.columns[0]
)
resume219_ids = resume219_tuning[resume219_idcol].drop_duplicates().tolist()[:100]
print("First 100 tuning stay IDs:", resume219_ids)
print("Current edits:", edited())

print("=== RETAINED CALIBRATION OBJECTS ===")
for k, v in list(globals().items()):
    if any(s in k.lower() for s in ["158", "193", "backup218", "paired", "causal", "target217"]):
        shape = getattr(v, "shape", None)
        print(k, type(v).__name__, tuple(shape) if shape is not None else "")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 219: recover after candidate158 factual validation\n"
        "- Forward 15 factual delta: +0.00002 nats/reading on 100 tuning stays, comfortably within the limit.\n"
        "- Candidate158 remains submitted; its dose response still requires official rollout validation.\n"
        "- Recovering architecture, schema, and tuning IDs before the next forward request.\n"
        "- Python runs: 219; forward passes: 15.\n"
    )
```