The latent projection is substantial, so a nearly zero `z_in` does not explain the failed posterior edits. I’ll instead test a direct value-head correction using the aligned original and fivefold-dose states. The fit will target the record-derived drug responses and penalize changes on factual inputs. Held-out dose sensitivity with a factual NLL increase below 0.005 would support this approach; failure to achieve both would refute it.

```python
import numpy as np
import pandas as pd
from scipy.special import softmax
from pathlib import Path

eh_X0 = np.asarray(r164_h0["phi"], dtype=np.float64)
eh_X5 = np.asarray(r164_h5["phi"], dtype=np.float64)
eh_D = eh_X5 - eh_X0
eh_rows = r164_h0["rows"].reset_index(drop=True)
eh_L0 = np.asarray(r164_h0["logits"], dtype=np.float64)
eh_L5 = np.asarray(r164_h5["logits"], dtype=np.float64)
eh_C = np.asarray(r164_h0["centres"], dtype=np.float64)
eh_y = np.asarray(r164_h0["truebin"], dtype=int)
eh_P0, eh_P5 = softmax(eh_L0, axis=1), softmax(eh_L5, axis=1)
eh_mu0 = (eh_P0 * eh_C).sum(1)
eh_mu5 = (eh_P5 * eh_C).sum(1)
eh_v = np.r_[0., (np.arange(1, 17) - 8.5) / 4.]
eh_deriv = (eh_P0 * (eh_C - eh_mu0[:, None]) * eh_v[None, :]).sum(1)

eh_kernels = {
    "insulin": np.array([-1.850, -2.256, -1.943, -1.561, -1.022, -0.690]),
    "norepinephrine": np.array([3.351, 2.621, 1.717, 0.943, 0.494, 0.409])
}
eh_target = np.zeros(len(eh_rows))
for sid, indices in eh_rows.groupby("stay_id").groups.items():
    rr = r164_records[r164_records.stay_id == sid]
    for drug, channels, median in (
        ("insulin", ("glucose_finger", "glucose_lab"), 1.5),
        ("norepinephrine", ("map_arterial", "map_cuff"), .46)
    ):
        dr = rr[(rr.event == drug) & rr.rate.notna()]
        if len(dr) == 0:
            continue
        dh = dr.hour.to_numpy(float)
        delta = 4. * (dr.rate.to_numpy(float) - median)
        for i in indices:
            if eh_rows.at[i, "event"] not in channels:
                continue
            ages = float(eh_rows.at[i, "hour"]) - dh
            valid = (ages > 1e-5) & (ages < 6.)
            lag = np.floor(ages[valid]).astype(int)
            eh_target[i] = np.dot(delta[valid], eh_kernels[drug][lag])

eh_t = (eh_target - (eh_mu5 - eh_mu0)) / np.maximum(eh_deriv, 1.)
eh_sids = np.sort(eh_rows.stay_id.unique())
eh_hold_ids = set(eh_sids[::3])
eh_hold = eh_rows.stay_id.isin(eh_hold_ids).to_numpy()
eh_fit = ~eh_hold
eh_mean = eh_X0[eh_fit].mean(0)
eh_F = np.column_stack([eh_X0 - eh_mean, np.ones(len(eh_rows))])
eh_DC = np.column_stack([eh_D, np.zeros(len(eh_rows))])

Gf = eh_F[eh_fit].T @ eh_F[eh_fit] / eh_fit.sum()
Gd = eh_DC[eh_fit].T @ eh_DC[eh_fit] / eh_fit.sum()
rhs = eh_DC[eh_fit].T @ eh_t[eh_fit] / eh_fit.sum()
ridge = 1e-4 * np.eye(eh_F.shape[1])
eh_fits = {}
base_nll = -np.log(eh_P0[np.arange(len(eh_y)), eh_y] + 1e-30)
print("ROWS", len(eh_rows), "held-out stays", len(eh_hold_ids))
print("FEATURE contrast RMS", float(np.sqrt(np.mean(eh_D**2))))
print("TARGET ranges", np.quantile(eh_target, [.01, .1, .5, .9, .99]))
print("lambda  factual_delta_hold  contrast_slope_hold  contrast_rmse_hold  weight_norm")
for lam in (0.03, .1, .3, 1., 3., 10., 30., 100., 300.):
    coef = np.linalg.solve(Gd + lam * Gf + ridge, rhs)
    eh_fits[lam] = coef
    factual_tilt = eh_F @ coef
    contrast_tilt = eh_DC @ coef
    pp = softmax(eh_L0 + factual_tilt[:, None] * eh_v, axis=1)
    nll = -np.log(pp[np.arange(len(eh_y)), eh_y] + 1e-30)
    slope = np.dot(contrast_tilt[eh_hold], eh_t[eh_hold]) / max(
        np.dot(eh_t[eh_hold], eh_t[eh_hold]), 1e-12)
    rmse = np.sqrt(np.mean((contrast_tilt[eh_hold] - eh_t[eh_hold])**2))
    print(lam, round(float((nll-base_nll)[eh_hold].mean()), 6),
          round(float(slope), 4), round(float(rmse), 4),
          round(float(np.linalg.norm(coef)), 3))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 193\nz_in has substantial weights; weak transmission cannot be blamed on a zero projection. Fitted closed-form value-head tilt candidates against original/5x matched-state contrasts, with factual-zero penalties and stay-held-out checks. Targets use recovered six-lag record-derived kernels and centered numeric dose differences. No edit installed.\n")
```
