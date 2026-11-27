The saved record analyses suggest a positive norepinephrine effect, but the current edit captures only part of it. I’ll preserve this validated candidate and inspect the fitting and calibration results. A stronger correction is worth testing only if held-out factual evidence—not merely training fit—supports the extra loss budget.

```python
import inspect
import numpy as np
import pandas as pd
from pathlib import Path

best_combined_v1 = {
    n: np.asarray(load(n), dtype=np.float32).copy()
    for n in head177_names
}
np.savez("/work/best_combined_v1.npz", **best_combined_v1)

print("MAP RAW FRONTIER")
print(map164_frontier.to_string(index=False))
print("\nMAP FIT KEYS:", list(map164_fits.keys()))
print("CURRENT MAP COEF NORM:", np.linalg.norm(head177_coef))
print("MAP SENSOR FRONTIER")
print(sensor165_frontier.head(12).to_string(index=False))

print("\nFUNCTION DEFINITIONS")
for name in ["head177_make", "head177_eval", "map164_mean"]:
    fn = globals()[name]
    try:
        print(name, "\n", inspect.getsource(fn)[:2400])
    except Exception:
        print(name, "names:", fn.__code__.co_names,
              "constants:", repr(fn.__code__.co_consts)[:1800])

print("\nCALIBRATION OBJECTS")
for name in [
    "cal170_nll", "cal170_biases", "cal170_rows",
    "cal170_base_train", "cal170_base_tune",
    "map164_K", "map164_F", "map164_b_ne",
    "map164_C_ne", "map164_C_i", "map164_train_mask", "map164_tune_mask"
]:
    obj = globals().get(name)
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, "\n", obj.head(8).to_string(index=False))
    elif isinstance(obj, dict):
        print(name, {k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100])
                     for k, v in obj.items()})
    elif hasattr(obj, "shape"):
        print(name, obj.shape)
        if np.asarray(obj).size <= 80:
            print(np.asarray(obj))
    else:
        print(name, repr(obj)[:800])

print("\nCAUSAL SENSOR ESTIMATES")
print(pd.read_csv("/work/sensor_iv.csv").to_string(index=False))

print("\nSELECTED NOTE SECTIONS")
sections_audit = Path("/work/notes.md").read_text().split("\n## ")
selected_audit = [
    s for s in sections_audit
    if any(w in s.lower() for w in ["calibrat", "embedding edit", "action edit", "proxy iv"])
]
for section in selected_audit[-5:]:
    print(section[:650])
```