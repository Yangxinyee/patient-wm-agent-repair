The last run confirmed that the flow-change candidate was saved, but it did not test its predictions. I’ll recover the architecture and tuning stay IDs, then check factual fit before spending a pass on rollouts. A factual NLL increase above 0.005 would rule this candidate out.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

_tuning_probe = pd.read_csv("/data/tuning.csv")
print("\n=== RECORD COLUMNS ===")
print(_tuning_probe.columns.tolist())
_stay_columns = [c for c in _tuning_probe.columns
                 if "stay" in c.lower() and "hour" not in c.lower()]
print("Stay columns:", _stay_columns)
if _stay_columns:
    _flow_test_ids = _tuning_probe[_stay_columns[0]].drop_duplicates().iloc[:100].tolist()
    print("Factual-test stay IDs:", _flow_test_ids)

print("\n=== CURRENT OPERATIONS ===")
print(ops())

print("\n=== PERSISTENT ARRAY / TABLE VARIABLES ===")
for _name, _value in list(globals().items()):
    if not _name.startswith("_") and isinstance(_value, (np.ndarray, pd.DataFrame)):
        print(_name, type(_value).__name__, _value.shape)

with open("/work/notes.md", "a") as _notes:
    _notes.write(
        "\n## Restart / Python run 62\n"
        "- Current candidate remains numeric gain 4 plus core.flow.a x1.15.\n"
        "- Recovering architecture and tuning IDs before factual validation; no edits changed.\n"
        "- Forward passes used: 24.\n"
    )
```