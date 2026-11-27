The recovered features include the prior variation that earlier unsafe candidates missed. I’ll fit a small ridge regression for MAP, constraining factual shifts and testing whether suppressing insulin-to-MAP spillover is costly. I’ll only consider fits that improve norepinephrine sensitivity and leave enough factual-loss headroom.

```python
import numpy as np, pandas as pd, torch
from scipy.linalg import solve

map164_mask = ~np.asarray(wm_glucose152, dtype=bool)
map164_tune_mask = ~np.asarray(wm_tune_glucose152, dtype=bool)
map164_train_mask = np.asarray(wm_split151, dtype=bool)
map164_X = wm_train145["inputs"].numpy().astype(np.float64)
map164_scale = np.asarray(wm_s152, dtype=np.float64)
map164_F = np.concatenate(
    [map164_X / map164_scale, np.ones((*map164_X.shape[:2], 1))], axis=2
)
map164_fact = map164_F[:, map164_mask & map164_train_mask].reshape(-1, 193)
map164_K = map164_fact.T @ map164_fact / len(map164_fact)
map164_C_ne = np.zeros((193, 193))
map164_b_ne = np.zeros(193)
map164_C_i = np.zeros((193, 193))

for _key, _cf in wm_cf146.items():
    _mask = (np.asarray(_cf["active"], dtype=bool)
             & map164_mask & map164_train_mask)
    _D = ((_cf["inputs"].numpy().astype(np.float64) - map164_X)
          [:, _mask] / map164_scale).reshape(-1, 192)
    _D = np.column_stack([_D, np.zeros(len(_D))])
    _C = _D.T @ _D / len(_D) / 2
    if _key[0] == "norepinephrine":
        _y = np.tile(np.asarray(wm_tau_targets151[_key])[_mask],
                     map164_X.shape[0])
        map164_C_ne += _C
        map164_b_ne += _D.T @ _y / len(_D) / 2
    else:
        map164_C_i += _C

def map164_mean(mc, raw_coef, mask, centres):
    w = torch.as_tensor(raw_coef, dtype=torch.float32)
    tau = (mc["inputs"] @ w[:-1] + w[-1]) * torch.as_tensor(mask)[None, :]
    p = wm_probs151(mc["logits"], tau)
    return (p.mean(0) * centres).sum(-1).numpy()

print("Insulin coefficient scaling check:",
      np.sqrt(np.mean((trial158_coef - trial158_gain *
                       np.r_[wm_fits152[(1.0, 0.0001)][:192] / wm_s152,
                             wm_fits152[(1.0, 0.0001)][192]])**2)))

map164_fits, map164_rows = {}, []
for _cross in [0.0, 0.3]:
    for _fw in [0.1, 0.3, 1.0, 3.0, 10.0]:
        for _ridge in [1e-4, 1e-3]:
            _R = np.eye(193) * _ridge
            _R[-1, -1] = 1e-8
            _scaled = solve(map164_C_ne + _cross * map164_C_i
                            + _fw * map164_K + _R,
                            map164_b_ne, assume_a="pos")
            _w = np.r_[_scaled[:192] / map164_scale, _scaled[192]]
            map164_fits[(_cross, _fw, _ridge)] = _w
            _row = {"cross": _cross, "fw": _fw, "ridge": _ridge,
                    "weight_norm": np.linalg.norm(_w)}
            for _label, _mc, _mask in [
                ("train", wm_train145, map164_mask),
                ("tuning", wm_tune145, map164_tune_mask),
            ]:
                _nll, _mu, _tau, _p = wm_rawloss152(_mc, _w, _mask)
                _dnll = (_nll - _mc["nll"]).mean(0).numpy()
                if _label == "train":
                    _row["cv_dNLL"] = float(_dnll[~map164_train_mask].mean())
                else:
                    _row["tuning_dNLL"] = float(_dnll.mean())
                    _row["tuning_tilt_rms"] = float(
                        torch.sqrt((_tau**2).mean()))
                    _row["latent_tilt_sd"] = float(
                        _tau[:, _mask].std(0).square().mean().sqrt())
            _base = map164_mean(wm_train145, _w, map164_mask,
                                wm_train145["centres"])
            for _key, _cf in wm_cf146.items():
                _mu = map164_mean(_cf, _w, map164_mask,
                                  wm_train145["centres"])
                _mask = (np.asarray(_cf["active"], dtype=bool)
                         & (wm_events151 == "map_arterial"))
                _dr = np.asarray(_cf["delta_rate"])[_mask]
                _shift = (_mu - _base)[_mask]
                _tag = ("N" if _key[0] == "norepinephrine" else "I_to_M")
                _row[f"{_tag}{_key[1]}"] = float(
                    np.dot(_dr, _shift) / np.dot(_dr, _dr))
            map164_rows.append(_row)

map164_frontier = pd.DataFrame(map164_rows)
map164_frontier.to_csv("/work/map_raw_frontier.csv", index=False)
print(map164_frontier.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 164: MAP-only stochastic ridge frontier\n"
             "- Fitted raw-head-input scalar tilts using NE local contrasts, "
             "all factual prior samples, and optional insulin spillover penalty.\n"
             "- Evaluated held-out training stays and tuning states. "
             "No new edits installed; insulin fallback remains active.\n")
```