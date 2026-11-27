Strength 2 is restored. The earlier norepinephrine decoder did not generalize, so I won’t use it to justify a stronger effect. I’ll recover the normalization and state alignment needed for a controlled numerical-dose probe. A reproducible dose-sensitive contrast could support a new correction; noisy or negligible contrasts would weaken that approach.

```python
import numpy as np
import pandas as pd

print("=== NUMERIC NORMALIZATION ===")
num118 = VOCAB["numeric"]
print("numeric type", type(num118).__name__)
if isinstance(num118, dict):
    print("numeric keys", list(num118))
    for k118, v118 in num118.items():
        if isinstance(v118, dict):
            print("field", k118, "keys", list(v118)[:20])
            for c118 in [8, 9, "8", "9", "insulin", "norepinephrine"]:
                if c118 in v118:
                    print(" ", c118, repr(v118[c118])[:1600])
        else:
            print(k118, repr(v118)[:700])
else:
    print(repr(num118)[:3500])

print("\n=== RAW NE DOSE/RATE RELATION ===")
raw118 = pd.read_csv(
    "/data/train.csv", nrows=15000,
    usecols=["stay_id", "hour", "event", "rate", "dose"])
ne118 = raw118[(raw118.event == "norepinephrine") & (raw118.rate > 0)]
print(ne118.head(4).to_string(index=False))
print((ne118.dose / ne118.rate).describe().to_string())

print("\n=== HEAD METADATA AND CONFIGURATION ===")
for k118 in ["meta96", "rh_meta", "head_meta"]:
    v118 = globals()[k118]
    print(k118, type(v118).__name__, getattr(v118, "shape", None))
    if isinstance(v118, pd.DataFrame):
        print("columns", v118.columns.tolist())
        print(v118.head(2).to_string(index=False))
for k118 in ["rh_cfg", "rh_rcfg", "rh_chosen", "rh_best", "candidate"]:
    print(k118, repr(globals()[k118])[:450])
for k118 in sorted(list(globals())):
    if k118.endswith(("96", "97", "98")):
        v118 = globals()[k118]
        if hasattr(v118, "shape"):
            print(k118, type(v118).__name__, tuple(v118.shape))
print("states x", states0["x"].shape)
print("state keys shapes",
      {k: getattr(v, "shape", None) for k, v in states0.items()})
```
