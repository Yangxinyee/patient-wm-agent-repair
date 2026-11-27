This check exposed another scaling inconsistency: the coefficients I assembled do not reproduce the previously validated decoder. I will not use them for an edit. I’ll refit directly on the verified physical action vectors and separately check contextual dose prediction. Low held-out errors are required before proceeding; large errors would rule out the gated-correction approach.

```python
import numpy as np
import torch

# Fresh regressions on physical action vectors; no model weights are changed.
repair_X = np.asarray(r41_X, dtype=np.float64)
repair_Y = np.asarray(r41_Y, dtype=np.float64)
repair_fit = np.asarray(r41_fitmask, dtype=bool)
repair_weights = np.asarray(r41_w, dtype=np.float64)
repair_scale = np.maximum(repair_X[repair_fit].std(axis=0), 0.02)
repair_Xs = repair_X / repair_scale
repair_sw = np.sqrt(repair_weights[repair_fit])
repair_A = repair_Xs[repair_fit] * repair_sw[:, None]
repair_B = repair_Y[repair_fit] * repair_sw[:, None]
repair_gram = repair_A.T @ repair_A
repair_rhs = repair_A.T @ repair_B
repair_ev, repair_Q = np.linalg.eigh(repair_gram)
repair_ev = np.maximum(repair_ev, 0)
repair_choices = []

print("=== RAW ACTION RIDGE VALIDATION ===")
for lam in (1e-7, 1e-5, 1e-3, 0.1, 1., 10., 100.):
    coef_s = repair_Q @ ((repair_Q.T @ repair_rhs) /
                         (repair_ev[:, None] + lam))
    coef_raw = coef_s / repair_scale[:, None]
    phat = repair_X[~repair_fit] @ coef_raw
    err = np.sqrt(np.mean((phat - repair_Y[~repair_fit])**2, axis=0))
    score = float(np.mean(err / np.array([3., .5, 1., 1.])))
    repair_choices.append((score, lam, coef_raw.copy(), err.copy()))
    print("ridge", lam, "RMSE [insulin, NE, presence I, presence NE]:",
          np.round(err, 5))
repair_choices.sort(key=lambda z: z[0])
repair_action_w = repair_choices[0][2].T.copy()
print("selected ridge:", repair_choices[0][1])

print("\n=== FACTUAL DOSE AND GATE CHECKS ===")
repair_actual = np.asarray(r47_actual_u, dtype=np.float64)
repair_actual_pred = repair_actual @ repair_action_w.T
repair_context = np.asarray(r41_ctx, dtype=np.float64)
repair_policy_w = {}
for j, drug in enumerate(r41_drug_names):
    on = r41_present[:, j] > .5
    old_w, old_b = r41_policies[drug]
    old_pred = repair_context @ old_w + old_b
    print(drug, "fresh dose RMSE:",
          float(np.sqrt(np.mean((repair_actual_pred[on,j]-r41_rates[on,j])**2))),
          "on-gate quantiles:",
          np.round(np.quantile(repair_actual_pred[on,j+2], [.01,.5,.99]), 4),
          "off-gate quantiles:",
          np.round(np.quantile(repair_actual_pred[~on,j+2], [.01,.5,.99]), 4))
    print("old policy predictions:", np.round(np.quantile(
        old_pred[on], [.01,.5,.99]), 4),
        "policy RMSE:", float(np.sqrt(np.mean(
            (old_pred[on]-r41_rates[on,j])**2))))
    repair_policy_w[drug] = (np.asarray(old_w).copy(), float(old_b))

print("\n=== RESPONSE-DIRECTION AND REMOVAL CHECKS ===")
print("chosen units:", r46_units,
      "chosen scores:", np.round(r46_score[r46_units], 6))
print("glucose z direction norm:", np.linalg.norm(r46_vg),
      "MAP z direction norm:", np.linalg.norm(r46_vm),
      "cosine:", float(np.dot(r46_vg, r46_vm) /
                      (np.linalg.norm(r46_vg)*np.linalg.norm(r46_vm))))
print("original readouts:",
      {k: float(g.loc[g["reached"], "shift"].mean())
       for k,g in roll0.groupby("edit")})
print("record-derived targets:", matched_refs)

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 52; forwards 3. Refit fresh dose/presence regressions "
            "directly on verified physical action vectors after another retained-"
            "coefficient scaling inconsistency. Checking policy predictions "
            "and latent directions before any edit; no weights changed.\n")
```