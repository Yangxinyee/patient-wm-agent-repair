The latest rollout shows a much stronger insulin response and a modest improvement for norepinephrine. That is useful only if factual fit still meets the limit. Next I’ll check the paired factual results and recover the record-derived effect estimates; those checks will determine whether this candidate is worth retaining.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import json, re

print("=== ARCHITECTURE ===")
_arch269 = Path("/model/ARCHITECTURE.md").read_text()
print(_arch269)
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

_notes269 = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION NOTES ===")
print(_notes269[:7500])

print("\n=== FORWARD 34: FACTUAL VALIDATION ===")
for _p269 in sorted(Path("/work/fwd_34").glob("*")):
    print(_p269.name)
    if _p269.suffix == ".json":
        print(_p269.read_text()[:4000])
    elif _p269.suffix == ".csv":
        _d269 = pd.read_csv(_p269)
        print("rows:", len(_d269), "columns:", list(_d269.columns))
        _nc269 = [c for c in _d269 if "nll" in c.lower()]
        print("NLL means:", _d269[_nc269].mean(numeric_only=True).to_dict())
        print(_d269.head(2).to_string(index=False))

print("\n=== SAVED JOINT EDIT CONTRACTS ===")
for _f269 in ["joint236.npz", "joint268.npz"]:
    with np.load("/work/" + _f269, allow_pickle=False) as _a269:
        print(_f269)
        for _k269 in _a269.files:
            _v269 = _a269[_k269]
            print(_k269, _v269.shape,
                  _v269.tolist() if _v269.size <= 12 else
                  {"norm": float(np.linalg.norm(_v269)), "dtype": str(_v269.dtype)})

print("\n=== RETAINED RELEVANT VARIABLES ===")
print([k for k in globals() if not k.startswith("_") and
       any(s in k.lower() for s in
           ["target", "causal", "effect", "insulin", "joint", "ridge", "blend", "scale"])])

with Path("/work/notes.md").open("a") as _log269:
    _log269.write(
        "\n## Run 269: restart audit of joint268\n"
        "Forward35 tuning100 rollout: insulin2=-10.852, insulinhalf=5.236, "
        "NE2=.593, NEhalf=-.914. Reading architecture, record-derived identification, "
        "paired factual34 and saved joint contracts before accepting any edit. "
        "No weights changed. Python runs: 269; forwards: 35.\n"
    )
```