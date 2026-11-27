The architecture confirms that generated readings feed back into later predictions, so a symmetric dose channel need not produce symmetric rollout effects. The file inspection succeeded, but the helper output was truncated. I’ll extract only the previous validation IDs and calibration helpers; paired GPU tests can then isolate the new edit’s effect.

```python
import inspect
import json
import re
from pathlib import Path
import pandas as pd

r268_validation_ids = pd.read_csv(
    "/work/fwd_29/dose_edit_edited.csv", usecols=["stay_id"]
).stay_id.drop_duplicates().astype(int).tolist()
print("VALIDATION IDS", r268_validation_ids)
print("COUNT", len(r268_validation_ids))

r268_items = list(globals().items())
print("\nCALIBRATION FUNCTIONS")
for name, obj in r268_items:
    if callable(obj) and hasattr(obj, "__code__") and re.search(
        r"sym|pair|candidate|replay|screen|factual", name, re.I
    ):
        print(name, str(inspect.signature(obj))[:180])

print("\nRECENT RETAINED OBJECTS")
for name, obj in r268_items:
    if not re.search(r"^r26[4-6]|^backup266$|^r250_passing$", name):
        continue
    if isinstance(obj, dict):
        print(name, "dict", str(list(obj)[:12])[:250])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, (int, float, tuple, list)):
        print(name, str(obj)[:250])
    elif callable(obj):
        print(name, "callable")
    else:
        print(name, type(obj).__name__)

print("\nLATEST FACTUAL SUMMARY")
print(Path("/work/fwd_28/summary.json").read_text()[:1600])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 268\n"
            "Recovered exact prior GPU validation stay IDs and bounded calibration "
            "helper metadata. No weights changed. Run-266 symmetric pair awaits "
            "paired GPU factual and dose-edit validation.\n")
```