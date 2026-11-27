The local probe shows weak direct transmission: about −0.6 glucose units per insulin-rate unit and +0.25 MAP units per norepinephrine-rate unit when doubling doses. Those are below the record-based estimates. I’ll test whether a small value-head correction can recover the missing sensitivity while staying quiet on factual records. Held-out factual loss rising sharply would refute this approach.

```python
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.linalg import solve

repair_axis = np.r_[0.0, np.arange(1, 17) - 8.5]
repair_fH = head_recovery["hidden"].numpy().astype(np.float64)
repair_fP = head_recovery["probs"].numpy().astype(np.float64)
repair_fbins = head_meta.target_bin.to_numpy(dtype=int)
repair_sids = head_meta.stay_id.to_numpy()
repair_sid_list = np.unique(repair_sids)
repair_fold_map = {sid: j % 2 for j, sid in enumerate(repair_sid_list)}
repair_folds = np.array([repair_fold_map[sid] for sid in repair_sids])
repair_fvar = (repair_fP * repair_axis**2).sum(1) - (repair_fP * repair_axis).sum(1)**2
repair_fvar = np.maximum(repair_fvar, 0.01)

repair_DH_parts, repair_T_parts, repair_W_parts = [], [], []
repair_cf_index_parts, repair_cf_key_parts = [], []
repair_drug_betas = {"insulin": -2.30803, "norepinephrine": 3.55456}
repair_groups = {}

for key, cf in repair_readout_cf.items():
    drug, factor = key
    idx = np.flatnonzero(cf["active"])
    sensor_relevant = (
        np.isin(events[idx], ["glucose_finger", "glucose_lab"])
        if drug == "insulin"
        else np.isin(events[idx], ["map_arterial", "map_cuff"])
    )
    desired = repair_drug_betas[drug] * cf["delta_rate"][idx] * sensor_relevant
    p_mid = 0.5 * (repair_base_P[idx] + cf["probs"][idx])
    jac = (p_mid * repair_C17[idx] * repair_axis).sum(1) - (
        (p_mid * repair_C17[idx]).sum(1) * (p_mid * repair_axis).sum(1)
    )
    jac = np.maximum(jac, 0.1)
    target_tilt = (desired - (cf["mean"][idx] - repair_base_mu[idx])) / jac
    variance = (p_mid * repair_axis**2).sum(1) - (p_mid * repair_axis).sum(1)**2
    repair_DH_parts.append(cf["hidden"][idx] - repair_base_H[idx])
    repair_T_parts.append(target_tilt)
    repair_W_parts.append(np.maximum(variance, 0.01))
    repair_cf_index_parts.append(idx)
    repair_cf_key_parts.extend([key] * len(idx))
    repair_groups[key] = (idx, desired)

repair_DH = np.concatenate(repair_DH_parts)
repair_T = np.concatenate(repair_T_parts)
repair_W = np.concatenate(repair_W_parts)
repair_cf_indices = np.concatenate(repair_cf_index_parts)
repair_cf_folds = repair_folds[repair_cf_indices]

def repair_tilt(P, tau):
    lp = np.log(np.maximum(P, 1e-30)) + tau[:, None] * repair_axis
    return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))

repair_head_cv_rows = []
repair_head_cv_predictions = {}
for fw in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
    for alpha in [0.01, 1.0, 100.0]:
        f_tau = np.zeros(len(head_meta))
        b_tau = np.zeros(len(head_meta))
        cf_tau = {key: np.zeros(len(head_meta)) for key in repair_readout_cf}
        for held in [0, 1]:
            fit_f = repair_folds != held
            fit_c = repair_cf_folds != held
            mean_h = repair_fH[fit_f].mean(0)
            sd_h = np.maximum(repair_fH[fit_f].std(0), 0.02)
            Xf = (repair_fH[fit_f] - mean_h) / sd_h
            Xc = repair_DH[fit_c] / sd_h
            A = fw * (Xf.T @ (repair_fvar[fit_f, None] * Xf))
            A += Xc.T @ (repair_W[fit_c, None] * Xc)
            A += alpha * np.eye(Xf.shape[1])
            rhs = Xc.T @ (repair_W[fit_c] * repair_T[fit_c])
            coef = solve(A, rhs, assume_a="pos") / sd_h
            intercept = -mean_h @ coef
            test = repair_folds == held
            f_tau[test] = repair_fH[test] @ coef + intercept
            b_tau[test] = repair_base_H[test] @ coef + intercept
            for key, cf in repair_readout_cf.items():
                cf_tau[key][test] = cf["hidden"][test] @ coef + intercept
        fp_new = repair_tilt(repair_fP, f_tau)
        dnll = np.log(np.maximum(repair_fP[np.arange(len(f_tau)), repair_fbins], 1e-30))
        dnll -= np.log(np.maximum(fp_new[np.arange(len(f_tau)), repair_fbins], 1e-30))
        pb_new = repair_tilt(repair_base_P, b_tau)
        mb_new = (pb_new * repair_C17).sum(1)
        row = {"factual_weight": fw, "alpha": alpha,
               "cv_dNLL": dnll.mean(), "factual_tilt_rms": np.sqrt(np.mean(f_tau**2))}
        err = []
        for key, cf in repair_readout_cf.items():
            pc_new = repair_tilt(cf["probs"], cf_tau[key])
            mc_new = (pc_new * repair_C17).sum(1)
            idx, desired = repair_groups[key]
            err.extend((mc_new[idx] - mb_new[idx] - desired).tolist())
            primary = "glucose_finger" if key[0] == "insulin" else "map_arterial"
            m = cf["active"] & (events == primary)
            dd = cf["delta_rate"][m]
            label = ("I" if key[0] == "insulin" else "N") + str(key[1])
            row[label + "_slope"] = dd @ (mc_new[m] - mb_new[m]) / (dd @ dd)
        row["cf_error_rms"] = np.sqrt(np.mean(np.asarray(err)**2))
        repair_head_cv_rows.append(row)
        repair_head_cv_predictions[(fw, alpha)] = (f_tau, b_tau, cf_tau)

repair_head_cv = pd.DataFrame(repair_head_cv_rows)
print(repair_head_cv.round(5).to_string(index=False))
repair_head_cv.to_csv("/work/head_contrast_cv.csv", index=False)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 79: cross-stay value-head contrast feasibility\n"
            "- Original local doubled-dose slopes: insulin approximately -0.62; NE approximately +0.25.\n"
            "- Tested ridge decoders of paired action contrasts, penalising activation on factual readings.\n"
            "- Targets use record-derived impulse estimates; this does not establish their long-horizon magnitude.\n"
            + repair_head_cv.to_string(index=False) + "\n")
```