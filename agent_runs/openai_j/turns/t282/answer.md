The reduced edit passes the previously failing batch, but barely: factual NLL increased by 0.00488 nats against the 0.005 limit. I’ll review the saved evidence and edit construction before choosing the next validation. More consistent factual margins would support retaining it; another failure would require backing off further.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

_notes248 = Path("/work/notes.md").read_text()
print("\n=== EARLY EVIDENCE / IDENTIFICATION NOTES ===")
print(_notes248[:14000])

print("\n=== SAVED CAUSAL TARGETS ===")
for _fn248 in ["kernel_targets.csv", "sensor_iv.csv", "dynamic_iv.csv", "dml_results.csv"]:
    _p248 = Path("/work") / _fn248
    _df248 = pd.read_csv(_p248)
    print(_fn248, "shape", _df248.shape)
    print(_df248.head(16).to_string(index=False))

print("\n=== AVAILABLE HELPER FUNCTIONS ===")
print(sorted(k for k, v in list(globals().items())
             if not k.startswith("_") and callable(v)
             and getattr(v, "__module__", "") in ("__main__", None)))

print("\n=== RELEVANT LIVE OBJECTS ===")
for _k248, _v248 in sorted(list(globals().items())):
    if _k248.startswith("_"):
        continue
    if any(t in _k248.lower() for t in
           ["candidate", "coef", "batch", "ids", "frontier", "calib", "strength",
            "head", "contrast", "feature", "tune", "cache", "target"]):
        if isinstance(_v248, np.ndarray):
            print(_k248, "array", _v248.shape, _v248.dtype)
        elif isinstance(_v248, pd.DataFrame):
            print(_k248, "DataFrame", _v248.shape, list(_v248.columns)[:12])
        elif isinstance(_v248, dict):
            print(_k248, "dict keys", list(_v248)[:18])
        elif isinstance(_v248, (list, tuple)):
            print(_k248, type(_v248).__name__, "length", len(_v248), repr(_v248[:8])[:300])

print("\n=== CURRENT EDIT CHECKPOINT ===")
with np.load("/work/safer_candidate247.npz") as _ck248:
    print([(k, _ck248[k].shape) for k in _ck248.files])
print("Edited tensors:", edited())

with Path("/work/notes.md").open("a") as _fh248:
    _fh248.write(
        "\n## Run 248: factual safety audit\n"
        "- Forward 28 / fwd_29: safer candidate (insulin strength 0.90, MAP gain 7.0)"
        " has value dNLL +0.00488 on tuning stays 200–299; passes narrowly.\n"
        "- No weights changed. Recovering causal evidence and exact construction"
        " before assessing aggregate factual safety and rollout performance.\n"
        "- Counts entering this run: 247 Python runs, 28 forward passes.\n"
    )
```