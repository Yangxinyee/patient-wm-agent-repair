The extension improved norepinephrine shifts to **+0.888 / −1.284 mmHg**, while preserving insulin at **−11.072 / +5.775 mg/dL**. MAP still falls short of the earlier record-derived estimate, especially for doubling. I’ll inspect the causal evidence and saved fits before deciding whether a more selective correction is justified; simply increasing gain could exhaust the factual-loss budget.

```python
from pathlib import Path
import inspect
import re
import numpy as np
import pandas as pd

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward 19: modest MAP rollout improvement\n"
        "- Current combined edit: insulin x2 -11.072, half +5.775; NE x2 +0.888, half -1.284.\n"
        "- Original: -3.680/+2.018 and +0.465/-0.871. MAP improvement is real but incomplete.\n"
        "- Forward 18 factual delta +0.00392 leaves limited margin; investigating selective rather than blind amplification.\n"
        "- 19 forward passes used.\n"
    )

notes_audit = Path("/work/notes.md").read_text()
print("EARLY CAUSAL NOTES\n", notes_audit[:4500])

print("\nRECENT VARIABLE NAMES")
recent_names_audit = [
    n for n in globals()
    if not n.startswith("_") and re.search(r"16[4-9]|17[0-9]", n)
]
print(" ".join(sorted(recent_names_audit)))

print("\nRELEVANT FUNCTION SIGNATURES")
for name, obj in sorted(list(globals().items())):
    if callable(obj) and re.search(
        r"map|head|contrast|candidate|sensor|kernel|compress|gate", name, re.I
    ):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            pass

print("\nHEAD177 OBJECTS")
for name, obj in sorted(list(globals().items())):
    if "177" not in name:
        continue
    if isinstance(obj, dict):
        print(name, {
            k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:120])
            for k, v in obj.items()
        })
    elif hasattr(obj, "shape"):
        print(name, tuple(obj.shape))
    elif not callable(obj):
        print(name, repr(obj)[:250])

for filename in ["sensor_iv.csv", "kernel_targets.csv", "map_raw_frontier.csv"]:
    table_audit = pd.read_csv("/work/" + filename)
    print("\n", filename, table_audit.shape)
    print(table_audit.head(6).to_string(index=False))
```