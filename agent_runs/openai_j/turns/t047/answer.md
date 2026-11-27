Doubling the dose/rate input weights increased factual NLL by only **0.00052 nats**, within the 0.005 limit. That makes this probe worth testing, but does not show that its treatment effects improved. I’ll recover the stay IDs and earlier identification evidence, then run paired rollouts. Effects moving toward the record-based estimates would support the edit; unchanged or misleading effects would argue against it.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes_path = Path("/work/notes.md")
notes_text = notes_path.read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(notes_text[:-8000])

tuning_ids = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].drop_duplicates().tolist()
probe_ids = [int(x) for x in tuning_ids[:100]]
print("\n=== FIRST 100 TUNING STAY IDS ===")
print(probe_ids)
print("\n=== CURRENT EDITS ===")
print(edited())
print("\n=== SAVED ANALYSES ===")
for filename in ["kernel_targets.csv", "dynamic_iv.csv", "sensor_iv.csv", "proxy_iv.csv"]:
    path = Path("/work") / filename
    if path.exists():
        frame = pd.read_csv(path)
        print(filename, "shape:", frame.shape)
        print(frame.head(12).to_string(index=False))

with notes_path.open("a") as f:
    f.write(
        "\n## Forward 6 and run 40\n"
        "- Doubling embed.numeric.weight dose/rate columns: tuning first-100 "
        "original value NLL 1.63164; edited 1.63216; difference +0.00052 "
        "(4,751 readings), below +0.005 tolerance.\n"
        "- This establishes only factual feasibility, not causal improvement. "
        "Recovering exact IDs and identification evidence before paired rollouts.\n"
        "- Python runs: 40. Forward passes: 6. Probe edit remains active.\n"
    )
```