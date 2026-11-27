None of the first moment-fit candidates meets the factual-loss budget. They appear too sensitive to the small fitting sample, so no weights changed. I’ll repeat the regression with a factual-preservation penalty covering all cached training and tuning states, without using tuning outcomes in that fit. This hypothesis is supported only if the stronger doubling response transfers to held-out stays at acceptable NLL.

```python
import numpy as np
import pandas as pd
import scipy.linalg as sla

def factual_curvature218(mc, pred, mask):
    p = pred["probs"][:, mask]
    var = ((p * v208.square()).sum(-1)
           - (p * v208).sum(-1).square()).numpy()
    x = mc["inputs"][:, mask].numpy().astype(np.float64)
    x = (x - center208) / scale208
    x = np.concatenate([x, np.ones(x.shape[:-1] + (1,))], axis=-1)
    x = x.reshape(-1, 193)
    return (x.T @ (x * var.reshape(-1, 1))) / len(x)

q218 = 0.5 * (
    factual_curvature218(wm_train145, base217, mapmask208)
    + factual_curvature218(wm_tune145, tbase217, tunemap208))
qs218 = np.trace(q218) / 193
moment218_candidates, moment218_rows = {}, []

def screen_map218(key, total_coef):
    correction = total_coef - map192_cal
    nt, mt = scalar_eval208(
        wm_train145, base217, correction, mapmask208, cent208)
    nu, mu = scalar_eval208(
        wm_tune145, tbase217, correction, tunemap208, tcent208)
    row = dict(key=str(key),
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
    moment218_candidates[key] = total_coef.copy()
    moment218_rows.append(row)

for pattern, target in target217.items():
    for ridge in [0.1, 1.0, 10.0]:
        for weight in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
            h = q218 + ridge * qs218 * np.eye(193) + weight * (m217.T @ m217)
            cs = sla.solve(h, weight * (m217.T @ target), assume_a="pos")
            screen_map218(("pooled", pattern, ridge, weight),
                          map192_cal + raw_coef208(cs))

for r in moment217_table.nsmallest(8, "tune_dNLL").itertuples():
    k = (r.pattern, r.ridge, r.weight)
    for strength in [0.25, 0.5, 0.75]:
        screen_map218(("rescaled",) + k + (strength,),
                      map192_cal + strength *
                      (moment217_candidates[k] - map192_cal))

moment218_table = pd.DataFrame(moment218_rows)
moment218_table.to_csv("/work/map_pooled_moment_frontier.csv", index=False)
safe218 = moment218_table[
    (moment218_table.cv_dNLL < 0.0045)
    & (moment218_table.tune_dNLL < 0.0043)]
print("=== SAFE POOLED/RESCALED FRONTIER ===")
print(safe218.sort_values("A2.0", ascending=False).head(12).to_string(index=False))
print("\n=== PRIOR HIDDEN-HEAD CONTRAST RESULTS ===")
old218 = pd.read_csv("/work/head_contrast_cv.csv")
print("columns:", old218.columns.tolist())
print(old218.head(4).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 217–218: moment-fit screening\n"
            "- Initial moment frontier had no candidate within factual budget; none installed.\n"
            "- Retesting with pooled, label-free factual curvature from cached training/tuning states and reduced strengths.\n"
            "- Active edit remains forward-21 calibrated combined candidate.\n")
```