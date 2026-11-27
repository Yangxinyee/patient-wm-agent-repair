The MAP extension raised factual value loss by **0.00392 nats**, still within the 0.005 limit. This does not establish that its dose predictions improved. I’ll recover the rollout inputs and candidate settings, then test whether it improves norepinephrine responses without disrupting the validated insulin repair.

```python
from pathlib import Path
import re
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README")
print(Path("/data/README.md").read_text())

f18_audit = pd.read_csv("/work/fwd_18/factual_original.csv")
print("\nFORWARD 18 COLUMNS:", list(f18_audit.columns))
stay_col_audit = next(
    c for c in ["stay_id", "stay", "id"] if c in f18_audit.columns
)
rollout_ids_audit = f18_audit[stay_col_audit].drop_duplicates().tolist()
print("FORWARD 18 STAYS:", rollout_ids_audit)

print("\nRELEVANT PERSISTENT VARIABLES")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or not re.search(
        r"best|cand|map|raw|head|kernel|contrast|sensor|ridge|stoch|prior|gate|tuning", name, re.I
    ):
        continue
    if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict):
        print(name, "dict keys:", list(obj.keys())[:30])
    elif callable(obj):
        print(name, "callable")
    elif isinstance(obj, (str, int, float, tuple, list)):
        text = repr(obj)
        print(name, text[:350])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Restart audit, run 179\n"
        "- Forward 18: MAP extension factual delta +0.00392 on 100 tuning stays; within +0.005.\n"
        "- Causal benefit remains untested. Recovering architecture, rollout IDs and persistent candidate settings.\n"
        "- Current four value-head edits retained pending dose-rollout validation; 18 forward passes used.\n"
    )
```