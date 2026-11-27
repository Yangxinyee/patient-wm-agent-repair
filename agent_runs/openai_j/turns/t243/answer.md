The current edit strengthens the halved-dose response much more than the doubled-dose response, on both fitting and held-out stays. That supports trying an asymmetric correction. I’ll fit low-cost average-response constraints, comparing a balanced dose correction with one that preserves the current half-dose response while strengthening doubling. Held-out slopes and factual NLL will decide whether either is useful.

```python
import numpy as np
import pandas as pd
import torch
import scipy.linalg as sla

base217 = score215["calibration_only"][0]
tbase217 = score215["calibration_only"][1]
gfact217 = response_design208(wm_train145, base217)

# Expected factual log-loss curvature for a small, separate linear tilt model.
p217 = base217["probs"][:, fitmap208]
var217 = ((p217 * v208.square()).sum(-1)
          - (p217 * v208).sum(-1).square()).numpy()
xx217 = wm_train145["inputs"][:, fitmap208].numpy().astype(np.float64)
xs217 = (xx217 - center208) / scale208
xs217 = np.concatenate(
    [xs217, np.ones(xs217.shape[:-1] + (1,))], axis=-1)
flat217 = xs217.reshape(-1, 193)
q217 = (flat217.T @ (flat217 * var217.reshape(-1, 1))) / len(flat217)
qscale217 = np.trace(q217) / 193

rows217, targets217, asymmetric217, labels217 = [], [], [], []
for factor in [0.5, 2.0]:
    mc = wm_cf146[("norepinephrine", factor)]
    bp = local216[("calibration_only", factor)]
    gc = response_design208(mc, bp)
    dr = np.asarray(mc["delta_rate"])
    active = np.asarray(mc["active"], bool)
    for event in ["map_arterial", "map_cuff"]:
        mask = fitall208 & active & (events206 == event)
        dose = dr[mask].mean()
        old_slope = (bp["mean"][mask] - base217["mean"][mask]).mean() / dose
        current_slope = (
            local216[("current", factor)]["mean"][mask]
            - score215["current"][0]["mean"][mask]).mean() / dose
        rows217.append((gc[mask] - gfact217[mask]).mean(0) / dose)
        targets217.append(beta208 - old_slope)
        asymmetric217.append(
            (current_slope if factor == 0.5 else beta208) - old_slope)
        labels217.append((factor, event))
m217 = np.asarray(rows217)
target217 = {
    "balanced": np.asarray(targets217),
    "double_focus": np.asarray(asymmetric217)
}
print("Moment labels:", labels217)
print("Moment matrix singular values:", np.linalg.svd(m217, compute_uv=False))
print("Factual curvature scale:", qscale217)

# Verify that scalar screening agrees with actual installed-head arithmetic.
z217 = map192_candidates[(10.0, 6.0)] - map192_cal
nn217, mm217 = scalar_eval208(
    wm_train145, base217, z217, mapmask208, cent208)
print("Scalar/head agreement:",
      np.max(np.abs(nn217 - score215["current"][0]["nll"])),
      np.max(np.abs(mm217 - score215["current"][0]["mean"])))

moment217_candidates = {}
moment217_rows = []
for pattern, target in target217.items():
    for ridge in [0.001, 0.01, 0.1]:
        for weight in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            h = q217 + ridge * qscale217 * np.eye(193) + weight * (m217.T @ m217)
            cs = sla.solve(h, weight * (m217.T @ target), assume_a="pos")
            correction = raw_coef208(cs)
            key = (pattern, ridge, weight)
            moment217_candidates[key] = map192_cal + correction
            nt, mt = scalar_eval208(
                wm_train145, base217, correction, mapmask208, cent208)
            nu, mu = scalar_eval208(
                wm_tune145, tbase217, correction, tunemap208, tcent208)
            row = dict(pattern=pattern, ridge=ridge, weight=weight,
                       fit_dNLL=float((nt-orig_train209)[fitall208].mean()),
                       cv_dNLL=float((nt-orig_train209)[cvall208].mean()),
                       tune_dNLL=float((nu-orig_tune209).mean()),
                       norm=float(np.linalg.norm(correction)))
            for factor in [0.5, 2.0]:
                mc = wm_cf146[("norepinephrine", factor)]
                _, cm = scalar_eval208(
                    mc, local216[("calibration_only", factor)],
                    correction, mapmask208, cent208)
                dr = np.asarray(mc["delta_rate"])
                for event, short in [("map_arterial", "A"), ("map_cuff", "C")]:
                    mask = cvall208 & np.asarray(mc["active"], bool) & (events206 == event)
                    row[f"{short}{factor}"] = float(
                        (cm[mask] - mt[mask]).mean() / dr[mask].mean())
            moment217_rows.append(row)

moment217_table = pd.DataFrame(moment217_rows)
moment217_table.to_csv("/work/map_moment_frontier.csv", index=False)
safe217 = moment217_table[
    (moment217_table.cv_dNLL < 0.0045)
    & (moment217_table.tune_dNLL < 0.0043)]
print("\n=== SAFE MOMENT FRONTIER, STRONGEST DOUBLING FIRST ===")
print(safe217.sort_values("A2.0", ascending=False).head(14).to_string(index=False))
print("\n=== FULL FRONTIER RANGE ===")
print(moment217_table[["cv_dNLL", "tune_dNLL", "A0.5", "A2.0"]].agg(["min", "max"]))
```