```python
from pathlib import Path
import re
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD COLUMNS ===")
print(Path("/data/README.md").read_text())

_recovery_notes = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK: RECORD-DERIVED EFFECTS ===")
print(_recovery_notes[:24000])

print("=== RELEVANT KERNEL VARIABLES ===")
print(sorted(k for k in globals()
             if not k.startswith("_")
             and re.search(r"caus|target|effect|coef|candidate|stay|cancel|baseline|truth",
                           k, re.I)))

print("=== FORWARD 43 FILES AND FORMAT ===")
print([p.name for p in Path("/work/fwd_43").iterdir()])
for _model in ("original", "edited"):
    _p = Path(f"/work/fwd_43/dose_edit_{_model}.csv")
    _d = pd.read_csv(_p)
    print(_model, _d.shape, _d.columns.tolist())
    print(_d.head(3).to_string(index=False))

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Forward 43 / Run 270: exact-checkpoint rollout recovery\n"
        "Checkpoint 864804d79e is unchanged. First100 tuning dose shifts: "
        "original insulin x2 -1.847, half +0.777; edited -12.958, +5.870. "
        "Original NE x2 -0.498, half -0.600; edited +0.948, -0.689. "
        "These are model predictions, not causal truth. All400 factual delta "
        "+0.00331143 (cluster upper95 +0.00486150) previously validated. "
        "Recovering training-record causal estimates before further decisions.\n"
    )
```
