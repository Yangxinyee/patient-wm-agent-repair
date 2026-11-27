```python
from pathlib import Path
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

header278 = pd.read_csv("/data/tuning.csv", nrows=2)
print("TUNING COLUMNS:", header278.columns.tolist())
idcol278 = next(c for c in ("stay_id", "stay", "icustay_id") if c in header278.columns)
tuning_ids278 = pd.read_csv("/data/tuning.csv", usecols=[idcol278])[idcol278].drop_duplicates().tolist()
print("FIRST 100 TUNING IDS:", tuning_ids278[:100])
print("CURRENT EDITS:", edited())
print("LATEST FORWARD FILES:", [p.name for p in Path("/work/fwd_35").iterdir()])
print("ACTION FRONTIER:")
print(pd.read_csv("/work/action_extrapolation_frontier273.csv").to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 278: restart validation setup\n"
            "- Six-tensor action-extrapolation candidate remains installed.\n"
            "- Recovering architecture and exact validation IDs before paired GPU validation.\n"
            "- Numerical preservation of factual action embeddings is not rollout validation.\n"
            "- Counts: 278 Python runs, 34 forward passes.\n")
```