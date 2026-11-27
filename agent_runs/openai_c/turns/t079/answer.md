Uniform query amplification fails the factual constraint by a wide margin: +0.09178 nats. I’ll remove it, but use its paired predictions as a controlled probe of head sensitivity. A selective correction is worth pursuing only if these changes can be predicted reliably from the saved states and concentrated in low-risk directions.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import inspect

q_probe_original = pd.read_csv("/work/fwd_23/factual_original.csv")
q_probe_edited = pd.read_csv("/work/fwd_23/factual_edited.csv")
q_probe_base = pd.read_csv("/work/fwd_9/factual_edited.csv")
assert q_probe_base[["stay_id", "hour", "event"]].equals(
    q_probe_edited[["stay_id", "hour", "event"]])

q_probe = q_probe_base[["stay_id", "hour", "event", "observed"]].copy()
q_probe["delta_mean"] = q_probe_edited["predicted_mean"] - q_probe_base["predicted_mean"]
q_probe["delta_nll"] = q_probe_edited["nll"] - q_probe_base["nll"]
print("QUERY-GAIN PAIRED PROBE")
print(q_probe.groupby("event").agg(
    n=("delta_mean", "size"),
    mean_shift=("delta_mean", "mean"),
    shift_sd=("delta_mean", "std"),
    mean_nll_change=("delta_nll", "mean")
).round(5).to_string())

clear_ops()
rank1("embed.numeric.weight", dose_numeric_direction, 3.0, side="in")
print("\nRESTORED", ops())

print("\nEXISTING STATE FILE SCHEMA")
with np.load("/work/fwd_13/states_edited.npz") as sf:
    for name in sf.files:
        a = sf[name]
        print(name, a.shape, a.dtype,
              a.reshape(-1)[:5].tolist() if a.size < 30000 else "large")

print("\nDOSE-MODEL KEYS")
for drug, item in dose_models.items():
    print(drug, list(item) if isinstance(item, dict) else type(item).__name__)
print("STATE COEFFICIENT HELPER", inspect.signature(state_dose_coefficients))
print("REMAINING VARIABLE NAMES",
      [n for n in sorted(globals()) if n.startswith(("s", "p", "t")) and not n.startswith("_")])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Forward 23 / Python run 54: uniform query gain rejected\n"
            "- Query gamma x1.5 plus numeric gain 4: factual NLL 1.72639 vs 1.63461, +0.09178.\n"
            "- Restored numeric gain 4. The paired query-only changes relative to forward 9 "
            "can be used as an empirical head-sensitivity probe, not as an accepted repair.\n"
            "- Next assess whether a selective head correction can be predicted and made factual-safe.\n"
            "- Forward passes used: 23.\n")
```
