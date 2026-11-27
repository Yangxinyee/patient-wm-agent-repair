The register test found a major diagnostic error: the presence/age/numeric layout reduced local NLL from 2.87 to 1.71. The old outcome-bin labels also disagree with freshly computed labels for 872 readings. I’ll correct both diagnostic inputs and average prior samples. Matching the paired edited-minus-original predictions would make local edit comparisons credible.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

head_ln = torch.stack(layouts231["presence_age_numeric6"], dim=1)
read_bins232 = labels231.copy()
last_bins232 = np.zeros(len(head_rows), dtype=np.int64)
for ev in head_rows.event.unique():
    ix = np.flatnonzero((head_rows.event.to_numpy() == ev) & head_rows.last_value.notna().to_numpy())
    last_bins232[ix] = np.searchsorted(
        num_spec[(ev, "value")]["fine_bin_edges"],
        head_rows.last_value.iloc[ix].to_numpy(), side="right") + 1
print("last-bin corrections:", int(np.sum(head_vb != last_bins232)))
head_vb = last_bins232.copy()

edit232 = np.load("/work/fwd_15/states_edited.npz")
rng232 = torch.Generator().manual_seed(232)
avg232 = {m: np.zeros((len(head_rows), 17), dtype=np.float64) for m in ["original", "edited"]}
for draw in range(16):
    noise = torch.randn((len(head_rows), 32), generator=rng232)
    for sign in [1, -1]:
        for model, weights, sf in [("original", WO225, orig231), ("edited", W225, edit232)]:
            probs, _, _, _ = head225(weights, sf, noise=noise * sign, status=None, typeidx=1)
            avg232[model] += probs.numpy() / 32

checks232 = {}
for model in ["original", "edited"]:
    gpu = pd.read_csv(f"/work/fwd_20/factual_{model}.csv")
    joined = head_rows[["stay_id", "hour", "event"]].assign(
        local_ix=np.arange(len(head_rows))).merge(gpu, on=["stay_id", "hour", "event"],
                                                  validate="one_to_one")
    ix = joined.local_ix.to_numpy()
    probs = avg232[model]
    nll = -np.log(np.maximum(probs[np.arange(len(probs)), read_bins232], 1e-30))
    mean = (probs * centres231).sum(1)
    checks232[model] = {"ix": ix, "nll": nll[ix], "mean": mean[ix],
                        "gpu_nll": joined.nll.to_numpy(), "gpu_mean": joined.predicted_mean.to_numpy()}
    print(model, "local/GPU NLL", round(float(nll[ix].mean()), 6),
          round(float(joined.nll.mean()), 6),
          "mean RMSE", round(float(np.sqrt(np.mean((mean[ix] - joined.predicted_mean.to_numpy())**2))), 6))

a, b = checks232["original"], checks232["edited"]
assert np.array_equal(a["ix"], b["ix"])
print("paired mean-shift RMSE", float(np.sqrt(np.mean(
    ((b["mean"] - a["mean"]) - (b["gpu_mean"] - a["gpu_mean"]))**2))))
print("paired NLL delta: local/GPU", float(np.mean(b["nll"] - a["nll"])),
      float(np.mean(b["gpu_nll"] - a["gpu_nll"])))

print("sensor_neurons:", sensor_neurons)
print("control-pair map:", _pair_for_control)
for ev, obj in head_audit.items():
    print("head audit", ev, {
        k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100])
        for k, v in obj.items()
    })

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 230–232: concrete diagnostic corrections\n"
            "- Correct register layout is consistent with [presence, age, z_value, has_value, "
            "z_dose, has_dose, z_rate, has_rate]; the recovered layout was wrong.\n"
            "- Recomputed outcome and last-value bins from each sensor's supplied edges.\n"
            "- No weight edits. Validating prior-averaged paired predictions before further calibration.\n")
```
