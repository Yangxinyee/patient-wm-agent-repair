The flexible kernels agree closely with the earlier six-lag estimates: insulin’s response is delayed and negative, while norepinephrine’s is positive and decays faster. I’ll apply these kernels to the exact rollout-eligible hours and check dose units. This will resolve whether the current insulin response is too small, too large, or mainly asymmetric.

```python
import numpy as np
import pandas as pd
from pathlib import Path

tuning238 = pd.read_csv("/data/tuning.csv")
roll238 = pd.read_csv("/work/fwd_23/dose_edit_edited.csv")
reach238 = roll238["reached"]
if reach238.dtype == object:
    reach238 = reach238.astype(str).str.lower().isin(["true", "1"])
else:
    reach238 = reach238.astype(bool)
roll238 = roll238[reach238 & roll238["hour"].between(13, 24)].copy()

print("=== RECORDED DOSE UNITS AND RANGES ===")
for drug238 in ["insulin", "norepinephrine"]:
    d238 = tuning238[tuning238["event"].eq(drug238)]
    print(drug238, "units", d238["rate_unit"].unique().tolist(),
          "rate quantiles", d238["rate"].quantile([0, .25, .5, .75, 1]).round(5).to_dict())

print("=== ORIGINAL ESTIMATE UNIT DEFINITIONS ===")
lines238 = Path("/work/notes.md").read_text().splitlines()
print("\n".join(lines238[72:85])[:1100])
print("make_panel constants:", repr(make_panel.__code__.co_consts)[:900])
print("cached targets:", target_table89.round(5).to_dict("records"))

dose238 = {}
for drug238 in ["insulin", "norepinephrine"]:
    d238 = tuning238[tuning238["event"].eq(drug238)].copy()
    d238["dose_hour"] = np.floor(d238["hour"].to_numpy() + 1e-6).astype(int)
    d238 = d238[d238["dose_hour"].between(12, 23)]
    for (stay238, hour238), rate238 in d238.groupby(["stay_id", "dose_hour"])["rate"].sum().items():
        dose238[(drug238, int(stay238), int(hour238))] = float(rate238)

matched_targets238 = {}
for edit238, drug238, factor238 in [
    ("insulin_x2", "insulin", 2.0), ("insulin_half", "insulin", .5),
    ("ne_x2", "norepinephrine", 2.0), ("ne_half", "norepinephrine", .5)
]:
    part238 = roll238[roll238["edit"].eq(edit238)].copy()
    exposure238 = np.zeros((len(part238), 12))
    for i238, (stay238, hour238) in enumerate(zip(part238["stay_id"], part238["hour"])):
        for lag238 in range(1, 13):
            exposure238[i238, lag238 - 1] = dose238.get(
                (drug238, int(stay238), int(hour238) - lag238), 0.0)
    kernel238 = np.asarray(kernels137[drug238], dtype=float)
    part238["kernel_shift_raw_rate"] = (factor238 - 1) * (exposure238 @ kernel238)
    matched_targets238[edit238] = {
        "frame": part238, "exposure": exposure238,
        "row_target": float(part238["kernel_shift_raw_rate"].mean()),
        "stay_target": float(part238.groupby("stay_id")["kernel_shift_raw_rate"].mean().mean())
    }
    print(edit238, "raw-rate kernel target",
          {k: round(v, 5) for k, v in matched_targets238[edit238].items()
           if isinstance(v, float)},
          "model row prediction", round(float(part238["shift"].mean()), 5))

with open("/work/notes.md", "a") as f:
    f.write("""
## Run 238
Applied retained 12-lag kernels to the exact fwd_23 reached-hour rows using
recorded doses from hour 12 onward. Checking raw-rate versus estimation
units before accepting matched targets. No saved weights changed.
""")
```
