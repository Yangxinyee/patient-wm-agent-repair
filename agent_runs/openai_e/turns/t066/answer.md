The new edit passes this factual check: NLL improved by 0.00048. That supports its selectivity, but not yet its intervention accuracy. I’ll recover the exact validation stays and earlier causal evidence before testing rollouts. Both insulin directions must approach the record-derived effects—not merely become larger.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())

factual59 = pd.read_csv("/work/fwd_6/factual_original.csv")
print("\nFACTUAL COLUMNS:", factual59.columns.tolist())
id_column59 = next(c for c in ["stay_id", "stay", "id"] if c in factual59.columns)
validation_ids59 = factual59[id_column59].drop_duplicates().astype(int).tolist()
print("VALIDATION STAYS:", validation_ids59)

notes59 = Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK")
print(notes59[:14000])

keys59 = [
    k for k in list(globals())
    if not k.startswith("_") and any(s in k.lower() for s in [
        "panel", "effect", "beta", "rho", "persist", "target", "trial",
        "prelude", "constrained", "state", "synth", "encode", "fit",
        "train", "tuning", "candidate", "direction"
    ])
]
print("\nRETAINED OBJECTS")
for k in keys59:
    obj59 = globals()[k]
    if isinstance(obj59, pd.DataFrame):
        desc59 = f"DataFrame {obj59.shape}, columns={obj59.columns.tolist()[:16]}"
    elif isinstance(obj59, np.ndarray):
        desc59 = f"array {obj59.shape}"
    elif isinstance(obj59, dict):
        desc59 = f"dict keys={list(obj59)[:12]}"
    elif callable(obj59):
        desc59 = f"callable {getattr(obj59, '__name__', '')}"
    elif isinstance(obj59, (int, float, str)):
        desc59 = repr(obj59)[:160]
    else:
        desc59 = type(obj59).__name__
    print(k, desc59)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 59: factual pass and restart recovery\n"
            "Forward 6 candidate factual NLL=1.64015 versus original=1.64063 "
            "(-0.00048) on 50 tuning stays. Admission-null insulin candidate "
            "passes this subset; intervention accuracy remains untested. "
            "Recovered architecture, validation IDs, causal notes and retained objects. "
            "No weight changes. Python runs: 59; forwards: 6.\n")
```
