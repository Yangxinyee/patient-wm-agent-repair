Using raw-rate convolution, the matched targets are −15.01/+7.50 for insulin and +4.87/−2.43 for norepinephrine—larger than the earlier targets. I won’t accept that discrepancy without checking units. I’ll compare the estimation panel with the source records and inspect the norepinephrine channel; agreement would validate these targets, while a hidden rescaling would invalidate them.

```python
import numpy as np
import pandas as pd
import torch
import dis

print("PANEL CONSTRUCTION CONSTANTS:", make_panel.__code__.co_consts)
print("LOCAL-PROJECTION CONSTANTS:", fit_lp136.__code__.co_consts)
print("PANEL COLUMNS:", list(pan.columns))

raw251 = pd.read_csv("/data/train.csv", nrows=8000,
                     usecols=["stay_id", "hour", "event", "rate"])
rawdrug251 = raw251[raw251["event"].isin(["insulin", "norepinephrine"])]
if len(rawdrug251):
    compare251 = rawdrug251.groupby(["stay_id", "hour", "event"])["rate"].sum().unstack()
    compare251 = compare251.rename(columns={
        "insulin": "raw_insulin", "norepinephrine": "raw_norepinephrine"
    }).reset_index()
    compare251 = compare251.merge(
        pan[["stay_id", "hour", "insulin", "norepinephrine"]],
        on=["stay_id", "hour"], how="inner"
    )
    print("RAW RATE VS ESTIMATION PANEL:")
    print(compare251.head(14).to_string(index=False))
    for drug in ("insulin", "norepinephrine"):
        use251 = compare251["raw_" + drug].notna() & (compare251["raw_" + drug] != 0)
        ratio251 = compare251.loc[use251, drug] / compare251.loc[use251, "raw_" + drug]
        print(drug, "panel/raw ratio:", ratio251.describe().round(6).to_dict())

print("DOSE-RESIDUAL OBJECTS:")
for drug in ("insulin", "norepinephrine"):
    obj251 = dml_residuals[drug]
    print(drug, "type", type(obj251).__name__)
    if isinstance(obj251, dict):
        for key, val in obj251.items():
            if isinstance(val, (float, int, str)):
                print(" ", key, val)
            elif hasattr(val, "shape"):
                a251 = np.asarray(val)
                print(" ", key, "shape", a251.shape,
                      "mean/std", float(np.nanmean(a251)), float(np.nanstd(a251)))
    print(" LP horizon-1 entry:", {
        key: (tuple(val.shape) if hasattr(val, "shape") and np.asarray(val).size > 10
              else np.asarray(val).round(6).tolist() if hasattr(val, "shape")
              else val)
        for key, val in lp_new136[drug][1].items()
        if not isinstance(val, dict)
    })

print("NE PAIRED CHANNEL:")
w251 = torch.as_tensor(load("z_post.0.weight"))
b251 = torch.as_tensor(load("z_post.0.bias"))
v251 = torch.as_tensor(load("z_post.2.weight"))
for pair251 in ((96, 184), (156, 89)):
    a251, g251 = pair251
    print(" rows", pair251,
          "biases", b251[list(pair251)].tolist(),
          "gate norm", float(w251[g251].norm()),
          "innovation norm", float((w251[a251] - w251[g251]).norm()),
          "output-column opposite error",
          float((v251[:, a251] + v251[:, g251]).norm()))
print("NE GLOBAL PARAMETER NAMES:",
      [n for n in globals() if any(s in n.lower() for s in
       ("amp_ne", "gain_ne", "gate_bias", "scale_ne", "rows_ne"))])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 251\n"
            "Run-250 reached-row raw-rate targets were insulin -15.009/+7.505, "
            "NE +4.866/-2.433; equal-stay targets differ. Auditing panel/raw-rate "
            "units and local-projection residual definitions before adopting them. "
            "Inspecting paired rows 156/89 for selective NE calibration. "
            "Current gain-1.25/gate-28 weights unchanged.\n")
```
