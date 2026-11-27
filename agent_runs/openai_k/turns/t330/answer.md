The stronger MAP trial increased factual NLL by about 0.00613 on these 100 stays, exceeding the allowed margin on this batch. It is not yet safe to retain. I’ll recover the exact candidates and validation batches; retaining this trial would require acceptable pooled factual loss and improved dose responses.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text()[:14000])

print("\nCANDIDATE ARCHIVES")
for filename in [
    "balanced_candidate.npz",
    "validated_current400.npz",
    "map_regularized_trial.npz",
    "glucose_115_candidate.npz",
]:
    with np.load("/work/" + filename, allow_pickle=False) as a:
        print(filename)
        for key in a.files:
            x = a[key]
            info = {"shape": x.shape, "dtype": str(x.dtype)}
            if x.size <= 12:
                info["value"] = x.tolist()
            elif np.issubdtype(x.dtype, np.number):
                info["norm"] = float(np.linalg.norm(x.astype(np.float64)))
            print(" ", key, info)

print("\nRETAINED VALIDATION/TARGET VARIABLES")
names = sorted(
    name for name in globals()
    if not name.startswith("_")
    and any(s in name.lower() for s in
            ["stay", "batch", "target", "convol", "effect", "eligible"])
)
for name in names[:100]:
    obj = globals()[name]
    if isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj[:8])[:220])
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:15])
    elif isinstance(obj, np.ndarray):
        print(name, "array", obj.shape, repr(obj[:8])[:220] if obj.ndim == 1 else "")
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns)[:15])
    elif isinstance(obj, (str, int, float, np.number)):
        print(name, repr(obj)[:250])
    else:
        print(name, type(obj).__name__)

print("\nEARLY IDENTIFICATION NOTES")
print(Path("/work/notes.md").read_text()[:6500])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 286: balanced-trial eligibility recovery\n"
        "- Forward 32 first-batch value-NLL delta approximately +0.00613; "
        "not eligible on that batch alone.\n"
        "- Recovering exact archives, architecture and validation inputs before "
        "pooled validation or rollback; no weights changed.\n"
        "- Python runs: 286; forward passes: 32.\n"
    )
```