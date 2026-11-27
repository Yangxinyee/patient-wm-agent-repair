The factual and counterfactual caches use the same corrected features and probabilities. I’ll fit a small value-head correction toward conservative immediate effects from the records, penalizing changes on factual readings. Held-out loss and dose-response checks will determine whether the fit is worth testing on the GPU.

```python
from scipy.special import logsumexp
from scipy.linalg import solve

r122_H = r121_np(verified_head["hidden"]).astype(np.float64)
r122_P = r117_base_probs.astype(np.float64)
r122_centres = r117_centres.astype(np.float64)
r122_bins = head_meta.target_bin.to_numpy(int)
r122_scale = head_meta.scale.to_numpy(float)
r122_basis = (np.arange(17, dtype=float) - 8.5) / 4
r122_basis[0] = 0.0
r122_Eb = r122_P @ r122_basis
r122_Vb = r122_P @ (r122_basis**2) - r122_Eb**2
r122_s0 = ((r122_P * r122_centres) @ r122_basis
           - r117_base_mean * r122_Eb)
r122_ids = np.sort(head_meta.stay_id.unique())
r122_fit = head_meta.stay_id.isin(r122_ids[:70]).to_numpy()
r122_test = ~r122_fit
r122_hmean = r122_H[r122_fit].mean(0)
r122_hsd = np.maximum(r122_H[r122_fit].std(0), 0.05)
r122_F = np.column_stack(((r122_H-r122_hmean)/r122_hsd,
                         np.ones(len(r122_H))))

def r122_tilt(P, tau):
    L = np.log(np.maximum(P, 1e-30)) + tau[:, None]*r122_basis
    return np.exp(L-logsumexp(L, axis=1)[:, None])

r122_parts = []
r122_cf_F = {}
for (drug, factor), cf in r117_cf.items():
    Hc = cf["hidden"].astype(float)
    Fc = np.column_stack(((Hc-r122_hmean)/r122_hsd, np.ones(len(Hc))))
    Pc = cf["probs"].astype(float)
    muc = cf["mean"]
    Ebc = Pc @ r122_basis
    sc = (Pc*r122_centres) @ r122_basis - muc*Ebc
    Xc = (sc[:, None]*Fc-r122_s0[:, None]*r122_F)/r122_scale[:, None]
    beta = np.where(head_meta.event.str.startswith("glucose"), -2.1, 0.0)
    if drug == "norepinephrine":
        beta = np.where(head_meta.event.str.startswith("map"), 3.4, 0.0)
    yc = (beta*cf["delta_rate"]-(muc-r117_base_mean))/r122_scale
    active = cf["active"] & (np.abs(cf["delta_rate"]) > 1e-7)
    r122_parts.append((Xc[active], yc[active], r122_fit[active],
                       2.0 if drug == "norepinephrine" else 1.0))
    r122_cf_F[(drug, factor)] = Fc

r122_X = np.concatenate([p[0] for p in r122_parts])
r122_y = np.concatenate([p[1] for p in r122_parts])
r122_cf_fit = np.concatenate([p[2] for p in r122_parts])
r122_cw = np.concatenate([np.full(len(p[1]), p[3]) for p in r122_parts])
r122_x = r122_X[r122_cf_fit]
r122_yf = r122_y[r122_cf_fit]
r122_wf = r122_cw[r122_cf_fit]
r122_Gc = (r122_x.T*r122_wf)@r122_x/r122_wf.sum()
r122_bc = r122_x.T@(r122_wf*r122_yf)/r122_wf.sum()
r122_Ff = r122_F[r122_fit]
r122_Gf = (r122_Ff.T*r122_Vb[r122_fit])@r122_Ff/r122_fit.sum()
r122_bf = r122_Ff.T@(r122_basis[r122_bins[r122_fit]]
                     - r122_Eb[r122_fit])/r122_fit.sum()
r122_models = {}
r122_results = []
for fw in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
    for ridge in [1e-5, 1e-4, 1e-3, 1e-2]:
        penalty = np.eye(r122_F.shape[1])*ridge
        penalty[-1, -1] *= 0.01
        coef = solve(r122_Gc + fw*r122_Gf + penalty,
                     r122_bc + fw*r122_bf, assume_a="pos")
        tau = r122_F@coef
        Pnew = r122_tilt(r122_P, tau)
        dnll = (np.log(np.maximum(r122_P[np.arange(len(Pnew)), r122_bins], 1e-30))
                - np.log(np.maximum(Pnew[np.arange(len(Pnew)), r122_bins], 1e-30)))
        rec = {"fw": fw, "ridge": ridge,
               "train_dNLL": dnll[r122_fit].mean(),
               "cv_dNLL": dnll[r122_test].mean(),
               "cv_tilt_rms": np.sqrt(np.mean(tau[r122_test]**2))}
        base_new_mu = (Pnew*r122_centres).sum(1)
        for (drug, factor), cf in r117_cf.items():
            Pcn = r122_tilt(cf["probs"], r122_cf_F[(drug, factor)]@coef)
            shift = (Pcn*r122_centres).sum(1)-base_new_mu
            sensor = "glucose_finger" if drug=="insulin" else "map_arterial"
            m = cf["active"] & r122_test & (head_meta.event.to_numpy()==sensor)
            dr = cf["delta_rate"][m]
            rec[f"{drug[0]}{factor}_slope"] = np.dot(dr, shift[m])/np.dot(dr, dr)
        r122_results.append(rec)
        r122_models[(fw, ridge)] = coef

r122_results = pd.DataFrame(r122_results)
print("Deterministic original NLL:",
      -np.log(np.maximum(r122_P[np.arange(len(r122_P)), r122_bins], 1e-30)).mean())
print("GPU original NLL:", head_meta.original_nll.mean())
print("Held-out stays:", r122_ids[70:].tolist())
print(r122_results.round(5).to_string(index=False))
print("Active edits:", edited())
```
