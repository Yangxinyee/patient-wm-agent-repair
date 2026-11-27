The architecture confirms that dose edits affect both numeric inputs and bins, and that generated readings feed back into later predictions. The output was truncated before the unit check. I’ll use a smaller inspection to recover the identification evidence and dose scale; this should tell us whether the retained targets are usable.

```python
from pathlib import Path
import numpy as np
import pandas as pd

g253 = dict(globals())
keys253 = sorted(n for n in g253 if any(
    t in n.lower() for t in
    ("kernel", "lp136", "rd_", "r244", "r250", "r246", "r249",
     "r233", "r238", "r243", "gate", "deconv", "causal")))
print("CALIBRATION OBJECT NAMES:", keys253)

print("\nPANEL UNITS")
for n in ("panel", "state_panel", "train_panel", "P", "panel136"):
    v = g253.get(n)
    if isinstance(v, pd.DataFrame):
        print(n, v.shape, "index:", v.index.names, "columns:", list(v.columns))
        if {"insulin", "norepinephrine"}.issubset(v.columns):
            print(v[["insulin", "norepinephrine"]].head(3).to_string())
            print(v[["insulin", "norepinephrine"]]
                  .agg(["min", "median", "max"]).round(6).to_string())

print("\nRAW RECORD SAMPLE")
print(pd.read_csv("/data/train.csv", nrows=4).to_string(index=False))
print("\nNUMERIC NORMALIZATION")
print({k: v for k, v in numeric_lookup_repair.items()
       if k[0] in ("insulin", "norepinephrine")})

text253 = Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK (first 5200 characters)")
print(text253[:5200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 253\n"
            "Run 252 architecture inspection succeeded, but output truncation hid the "
            "unit audit. Bounded inspection of dose normalization, calibration names, "
            "and early identification notes. No weights changed.\n")
```