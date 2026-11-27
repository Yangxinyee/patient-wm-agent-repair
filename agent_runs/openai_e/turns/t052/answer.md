The trial moves glucose in the intended direction locally, but its factual leakage and uneven dose responses make it unproven. I’ll recover the model specification and validation IDs, then test it on the GPU. A factual NLL increase above 0.005 would rule out this edit.

```python
from pathlib import Path
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())
print("\nEARLY NOTEBOOK")
print(Path("/work/notes.md").read_text()[:12000])

recovery_tuning = pd.read_csv("/data/tuning.csv")
print("\nTUNING COLUMNS:", recovery_tuning.columns.tolist())
print(recovery_tuning.head(3).to_string(index=False))
print("\nRETAINED OBJECTS")
for recovery_name, recovery_obj in list(globals().items()):
    if recovery_name.startswith("_") or recovery_name.startswith("recovery"):
        continue
    if isinstance(recovery_obj, pd.DataFrame):
        print(recovery_name, "DataFrame", recovery_obj.shape, recovery_obj.columns.tolist()[:18])
    elif callable(recovery_obj) and getattr(recovery_obj, "__module__", "") == "__main__":
        print(recovery_name, "function")
    elif isinstance(recovery_obj, dict) and len(recovery_obj) < 35:
        print(recovery_name, "dict keys", list(recovery_obj)[:20])
print("CURRENT EDITS:", edited())

with open("/work/notes.md", "a") as recovery_log:
    recovery_log.write(
        "\n## Run 48: restart and trial-validation setup\n"
        "The core.b_in.weight trial is not accepted: local direction is encouraging "
        "but factual leakage and asymmetric dose contrasts require GPU validation. "
        "Recovered architecture, early evidence and tuning identifiers. "
        "Python runs: 48; forward passes: 3.\n"
    )
```