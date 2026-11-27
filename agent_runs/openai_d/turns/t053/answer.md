Removing the register directions restores insulin innovation decoding (held-out R² ≈ 0.29), but norepinephrine remains essentially undecodable at the head. I’ll test an insulin-only head correction first, with a gate to keep it off other reading types. Reliable gating and a small factual perturbation would justify a forward trial.

```python
import torch
import torch.nn.functional as F

rec_head_X = np.asarray(repair_wc_mean, dtype=np.float64) @ rec_reg_projector
rec_head_groups = head_meta["stay_id"].to_numpy()
rec_finger = head_meta["event"].to_numpy() == "glucose_finger"
rec_gate_y = np.where(rec_finger, 1., -1.)
rec_gate_pred = np.zeros(len(rec_gate_y))
for rec_tr, rec_va in GroupKFold(4).split(rec_head_X, rec_gate_y, rec_head_groups):
    rec_fit = Ridge(alpha=.001).fit(rec_head_X[rec_tr], rec_gate_y[rec_tr])
    rec_gate_pred[rec_va] = rec_fit.predict(rec_head_X[rec_va],)
rec_gate_fit = Ridge(alpha=.001).fit(rec_head_X, rec_gate_y)
rec_gate_v = rec_reg_projector @ rec_gate_fit.coef_
rec_gate_b = float(rec_gate_fit.intercept_)
print("=== GLUCOSE-FINGER GATE ===")
print("held-out accuracy:", float(np.mean((rec_gate_pred > 0) == rec_finger)))
print("finger margins:", np.quantile(rec_gate_pred[rec_finger], [0, .01, .5, .99, 1]))
print("other margins:", np.quantile(rec_gate_pred[~rec_finger], [0, .01, .5, .99, 1]))

rec_ins_a = repair_aligned["insulin"]
rec_ins_target = np.nan_to_num(rec_ins_a["rd"].to_numpy(), nan=0.)
rec_ins_indices = np.flatnonzero(rec_finger)
rec_ins_cv = []
for rec_alpha in [.0001, .001, .01, .1, 1., 10.]:
    rec_pred = np.zeros(len(rec_ins_indices))
    rec_x = rec_head_X[rec_ins_indices]
    rec_y = rec_ins_target[rec_ins_indices]
    rec_g = rec_head_groups[rec_ins_indices]
    for rec_tr, rec_va in GroupKFold(4).split(rec_x, rec_y, rec_g):
        rec_fit = Ridge(alpha=rec_alpha).fit(rec_x[rec_tr], rec_y[rec_tr])
        rec_pred[rec_va] = rec_fit.predict(rec_x[rec_va])
    rec_ins_cv.append((rec_alpha, float(r2_score(rec_y, rec_pred)), rec_pred))
rec_ins_best = max(rec_ins_cv, key=lambda q: q[1])
rec_ins_fit = Ridge(alpha=rec_ins_best[0]).fit(
    rec_head_X[rec_ins_indices], rec_ins_target[rec_ins_indices])
rec_ins_v = rec_reg_projector @ rec_ins_fit.coef_
rec_ins_b = float(rec_ins_fit.intercept_)
print("\n=== INSULIN DECODER, INCLUDING INACTIVE HOURS ===")
print("best alpha / held-out R2:", rec_ins_best[:2])
print("held-out prediction quantiles:",
      np.quantile(rec_ins_best[2], [0, .01, .5, .99, 1]))
print("register contamination:",
      float(np.max(np.abs(rec_ln.T @ rec_ins_v))),
      float(np.max(np.abs(rec_vb @ rec_ins_v))))

print("\n=== RECONSTRUCTION TENSOR CACHES ===")
for rec_name, rec_obj in list(globals().items()):
    if (rec_name.startswith(("wr_", "rh_", "repair_"))
            and isinstance(rec_obj, torch.Tensor)):
        print(rec_name, tuple(rec_obj.shape))
print("chosen reconstruction config:", rh_chosen.get("config"))
print("vocab keys:", list(VOCAB))
print("numeric_specs insulin:", numeric_specs.get("insulin"))
print("vocab glucose centres:", VOCAB["value_bin_centres"].get("glucose_finger")
      if isinstance(VOCAB["value_bin_centres"], dict) else "not a dict")
print("Current edits:", edited())
```