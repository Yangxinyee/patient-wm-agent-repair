The architecture confirms that dose magnitude enters through the three edited tensors, consistent with the observed rollout invariance. The output was truncated before showing the cached analysis variables. I’ll retrieve those details so I can test late-hour effects using only pre-dose history; reproducible nonzero estimates would challenge the zero-effect hypothesis.

```python
import inspect, json
from pathlib import Path
import numpy as np
import pandas as pd

print("=== CACHED ARRAYS / TABLES ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if isinstance(obj, np.ndarray):
        print(name, obj.shape, str(obj.dtype))
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape)
    elif isinstance(obj, pd.Series):
        print(name, "Series", obj.shape, str(obj.dtype))
    elif isinstance(obj, dict) and name not in {"SHAPES", "VOCAB"}:
        print(name, "dict:", list(obj)[:12])
    elif isinstance(obj, (list, tuple)) and len(obj) <= 30:
        if all(isinstance(v, (str, int, float, np.integer)) for v in obj):
            print(name, repr(obj))

print("\n=== FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            print(name, inspect.signature(obj))
        except (TypeError, ValueError):
            pass

print("\n=== EDIT SUMMARY ===")
for name in edited():
    w, w0 = np.asarray(load(name)), np.asarray(load_original(name))
    print(name, w.shape, "max absolute change:", float(np.max(np.abs(w - w0))))

tuning_ids = np.sort(pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique())
print("\nNEXT FACTUAL REQUEST:")
print(json.dumps({"type": "factual", "model": "both", "split": "tuning",
                  "stays": tuning_ids[100:200].astype(int).tolist()}))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Restart architecture check\n"
            "Architecture re-read. Dose/rate magnitude enters the three edited embedding tensors. "
            "The first 100-stay factual comparison passed; all four paired rollouts were invariant. "
            "Still checking late-hour causal evidence and the remaining tuning stays.\n")
```
