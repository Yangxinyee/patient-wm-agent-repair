An isolated insulin increment lowers the next predicted glucose reading by about 1 mg/dL, but almost all of that state response disappears by the following hour. Norepinephrine barely responds. I’ll test the retained alternative connection directions for a stronger, stable insulin response. Better likelihood and greater response together would support one; runaway channel activity would reject it.

```python
import numpy as np, torch, pandas as pd, time
from pathlib import Path

decoder160 = torch.as_tensor(chosen_channel116["decoder"], dtype=torch.float32)
original_b160 = torch.as_tensor(load_original("core.b_in.weight"), dtype=torch.float32)
channel_weights160 = dict(weights159)
channel_weights160["core.b_in.weight"] = original_b160
original_weights160 = {n: torch.as_tensor(load_original(n), dtype=torch.float32) for n in NAMES}

original_x160, _ = replay_factual125(original_weights160)
_, original_p160 = head_features_fixed({"x": original_x160}, dataset159, draws=4, seed=16001)
bins160 = dataset159.observed_bin.to_numpy(dtype=int)
original_nll160 = -np.log(np.maximum(original_p160[np.arange(len(bins160)), bins160], 1e-30)).mean()
channel_x160, channel_q160 = replay_factual125(channel_weights160)
print("Original local factual NLL:", round(float(original_nll160), 6))
for label160, mask160 in [
    ("insulin", repair_bank["has_insulin"]),
    ("NE", repair_bank["has_norepinephrine"]),
    ("no drug", ~(repair_bank["has_insulin"] | repair_bank["has_norepinephrine"]))
]:
    vals160 = channel_q160[np.asarray(mask160, dtype=bool)]
    print("Channel-only q", label160, "mean/std/maxabs",
          np.round([np.nanmean(vals160), np.nanstd(vals160), np.nanmax(np.abs(vals160))], 5))

chosen160 = impulse_results159["insulin"]["chosen"]
ix160 = chosen160.bundle_index.to_numpy(dtype=int)
rates160 = chosen160.rate.to_numpy(dtype=float)
doses160 = chosen160.dose.to_numpy(dtype=float)
new_u160 = singleton_action_repair(atom_exact65(
    "insulin", np.full(len(ix160), np.nan), rates160 + 1,
    doses160 + doses160 / rates160))
impulse_hour160 = dict(zip(chosen160.stay_id, chosen160.hour))
lags160 = np.ceil(dataset159.hour.to_numpy()
                  - dataset159.stay_id.map(impulse_hour160).to_numpy() - 1e-4)
lag_masks160 = [(dataset159.event.to_numpy() == "glucose_finger") & (lags160 == lag)
               for lag in range(1, 5)]

connection_tests160 = {}
saved_u160 = U65.clone()
for ridge160 in [0.0001, 0.001, 0.01, 0.1]:
    direction160 = torch.as_tensor(directions118[ridge160], dtype=torch.float32)
    for scale160 in [0.5, 1.0]:
        ww160 = dict(channel_weights160)
        ww160["core.b_in.weight"] = original_b160 + scale160 * torch.outer(direction160, decoder160)
        start160 = time.time()
        xx160, qq160 = replay_factual125(ww160)
        if not np.isfinite(xx160).all() or np.nanmax(np.abs(qq160)) > 100:
            print(ridge160, scale160, "rejected: nonfinite or large channel activity")
            continue
        _, pp160 = head_features_fixed({"x": xx160}, dataset159, draws=4, seed=16001)
        nll160 = -np.log(np.maximum(pp160[np.arange(len(bins160)), bins160], 1e-30)).mean()
        factual_means160 = (pp160 * centres159).sum(axis=1)
        try:
            U65[ix160] = new_u160
            xi160, qi160 = replay_factual125(ww160)
        finally:
            U65[:] = saved_u160
        _, pi160 = head_features_fixed({"x": xi160}, dataset159, draws=4, seed=16001)
        delta160 = (pi160 * centres159).sum(axis=1) - factual_means160
        response160 = [float(np.mean(delta160[m])) for m in lag_masks160]
        summary160 = {
            "dNLL": float(nll160 - original_nll160),
            "state_rmse": float(np.sqrt(np.mean((xx160 - original_x160)**2))),
            "q_change_rmse": float(np.sqrt(np.nanmean((qq160 - channel_q160)**2))),
            "impulse_lags": response160,
            "direction_norm": float(torch.linalg.vector_norm(direction160))
        }
        connection_tests160[(ridge160, scale160)] = summary160
        print(ridge160, scale160,
              {k: np.round(v, 5).tolist() if isinstance(v, list) else round(v, 6)
               for k, v in summary160.items()},
              "seconds", round(time.time() - start160, 2))

Path("/work/notes.md").open("a").write(
    "\n## Run 160\nRun-159 factual replay showed insulin impulse responses about "
    "[-0.989,-0.148,-0.041,-0.015] at lags 1–4 versus sustained record kernels; NE responses near zero. "
    "Tested retained connection directions for factual stability and isolated-dose response. "
    "These are mechanism diagnostics, not counterfactual truth. No weights changed.\n")
```
