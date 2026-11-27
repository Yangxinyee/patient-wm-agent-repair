The installed edit produces the intended local reading shifts, but its factual attention perturbation is sizable; that is not a factual-fit result. I’ll select a representative tuning batch and check the exposure units used in the record regressions. The GPU test will decide whether this gain is acceptable.

```python
import numpy as np
import pandas as pd
from pathlib import Path

r248_ids = np.sort(r246_tuning["stay_id"].unique())
r248_batch = np.sort(np.random.default_rng(248).choice(r248_ids,size=100,replace=False)).astype(int).tolist()
print("FACTUAL TEST STAYS")
print(r248_batch)

print("\nDOSE UNITS")
for r248_drug in ("insulin","norepinephrine"):
    r248_sub = r246_tuning[r246_tuning["event"]==r248_drug]
    print(r248_drug,
          {c:r248_sub[c].dropna().unique().tolist() for c in ("rate_unit","dose_unit")},
          "rate quantiles",r248_sub["rate"].quantile([0,.5,.9,1]).round(5).tolist(),
          "dose quantiles",r248_sub["dose"].quantile([0,.5,.9,1]).round(5).tolist())

print("\nCAUSAL EXPOSURE CACHE")
if isinstance(causal_residuals,dict):
    for r248_key in list(causal_residuals)[:4]:
        r248_obj = causal_residuals[r248_key]
        print("key",r248_key,"type",type(r248_obj).__name__)
        if isinstance(r248_obj,(tuple,list)):
            for r248_j,r248_x in enumerate(r248_obj[:7]):
                r248_a = np.asarray(r248_x)
                if np.issubdtype(r248_a.dtype,np.number):
                    print(r248_j,r248_a.shape,"mean",round(float(np.nanmean(r248_a)),5),
                          "sd",round(float(np.nanstd(r248_a)),5))
        elif isinstance(r248_obj,dict):
            print("keys",list(r248_obj)[:12])

print("\nDOSE EXPORT SCHEMA")
r248_files = list(Path("/work/fwd_20").glob("dose_edit_*.csv"))
if r248_files:
    r248_export = pd.read_csv(r248_files[0])
    print(r248_export.columns.tolist())
    print(r248_export.head(3).to_string(index=False))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 248\nSelective candidate has intended local signs but held attention perturbation RMS4.07; this is not proof of factual safety. Selected random100 tuning stays for paired factual test and inspected causal-exposure units. No additional weights changed. Counts: 248 Python runs, 23 forwards.\n")
```