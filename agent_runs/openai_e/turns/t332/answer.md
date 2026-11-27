The stronger insulin edit worsened factual NLL by about **0.00672**, exceeding the 0.005 limit on this cohort. I won’t submit it as-is. Next I’ll inspect the saved safer edit and exact losses; a smaller adjustment is worthwhile only if it improves insulin responses while preserving factual fit.

```python
import os, json, numpy as np, pandas as pd

_r279_arch = open("/model/ARCHITECTURE.md").read()
print("ARCHITECTURE\n", _r279_arch)

print("\nSAVED CANDIDATE CONTRACTS")
for _r279_file in ["joint236.npz", "joint268.npz", "joint274.npz", "joint278.npz"]:
    with np.load("/work/" + _r279_file, allow_pickle=False) as _r279_z:
        print(_r279_file)
        for _r279_k in _r279_z.files:
            _r279_a = _r279_z[_r279_k]
            print(" ", _r279_k, _r279_a.shape,
                  _r279_a.tolist() if _r279_a.size <= 8 else
                  ("norm=" + str(float(np.linalg.norm(_r279_a)))
                   if np.issubdtype(_r279_a.dtype, np.number) else str(_r279_a.dtype)))

print("\nFORWARD40 EXACT FACTUAL SUMMARY")
for _r279_model in ["original", "edited"]:
    _r279_df = pd.read_csv("/work/fwd_40/factual_" + _r279_model + ".csv")
    print(_r279_model, "rows", len(_r279_df), "columns", list(_r279_df.columns))
    print({c: float(_r279_df[c].mean()) for c in _r279_df.columns
           if "nll" in c.lower() and pd.api.types.is_numeric_dtype(_r279_df[c])})

print("\nRELEVANT PERSISTENT VARIABLES")
_r279_names = [k for k in list(globals())
               if not k.startswith("_r279")
               and any(s in k.lower() for s in
                       ["joint", "blend", "insulin", "ne_scale", "target", "candidate", "delta_i", "delta_ne"])]
print(_r279_names[:100])
print("Current edits:", edited())

with open("/work/notes.md", "a") as _r279_f:
    _r279_f.write(
        "\n## Run 279: reject stronger insulin trial on measured cohort\n"
        "Forward40 joint278 factual delta approximately .00672 on tuning100..199, "
        "above .005. Not accepted. Recovering architecture, archived candidate "
        "contracts and exact loss before restoring fallback or constructing a "
        "smaller insulin-prioritized trial. No weights changed in this run.\n"
        "Python runs: 279; forwards: 40.\n"
    )
```