The constraints improve proportionality, but norepinephrine accuracy worsens and cross-drug leakage remains substantial. I won’t use those decoders. Next I’ll test a low-rank action-encoder edit that exposes each drug’s rate in four weakly used directions. It must recover held-out rates across x0–x3 while barely disturbing the original action representation.

```python
import numpy as np
import torch
from scipy.linalg import solve, null_space, eigh

def rr50_action_parts(insulin_factor=1.0, ne_factor=1.0):
    n = len(r36_indices)
    vectors, owners = [], []
    for owner, code, rate, dose in repair_action_atoms:
        factor = insulin_factor if code == "insulin" else ne_factor
        vectors.append(r36_atom(code, np.nan, rate * factor, dose * factor, False))
        owners.append(owner)
    vectors = torch.stack(vectors)
    owners = torch.tensor(owners, dtype=torch.long)
    counts = torch.bincount(owners, minlength=n).float()
    pooled = torch.zeros((n, 192), dtype=torch.float32)
    pooled.index_add_(0, owners, r33_mlp(vectors, "encode_action.mlp1", "gelu"))
    pooled /= counts.clamp_min(1).sqrt()[:, None]
    pooled += r33_linear(torch.log1p(counts)[:, None], "encode_action.size")
    hidden = r33_act(r33_linear(pooled, "encode_action.mlp2.0"), "gelu")
    pre = r33_linear(hidden, "encode_action.mlp2.2")
    return hidden.numpy(), pre.numpy(), counts.numpy()

rr50_cache = {("base", 1.0): rr50_action_parts()}
for _key in rr49_alts:
    _d, _factor = _key
    _factors = [1.0, 1.0]
    _factors[_d] = _factor
    rr50_cache[_key] = rr50_action_parts(*_factors)

rr50_scales = np.array([3.0, 0.6])
rr50_targets = {}
for _key, (_H, _pre, _counts) in rr50_cache.items():
    _rates = rr48_rates.copy()
    if _key[0] != "base":
        _rates[:, _key[0]] *= _key[1]
    _features = np.column_stack([_rates / rr50_scales, rr48_present.astype(float)])
    _sigma = np.sqrt(np.var(_pre.astype(np.float64), axis=1) + 1e-5)
    rr50_targets[_key] = (_features, _sigma)

rr50_fit_H, rr50_fit_Y, rr50_fit_pre = [], [], []
for _key, (_H, _pre, _counts) in rr50_cache.items():
    _use = rr49_train if _key[0] == "base" else (
        rr49_train & rr48_present.any(axis=1))
    _features, _sigma = rr50_targets[_key]
    rr50_fit_H.append(_H[_use])
    rr50_fit_Y.append(_features[_use] * _sigma[_use, None])
    rr50_fit_pre.append(_pre[_use])
rr50_fit_H = np.concatenate(rr50_fit_H).astype(np.float64)
rr50_fit_Y = np.concatenate(rr50_fit_Y).astype(np.float64)
rr50_fit_pre = np.concatenate(rr50_fit_pre).astype(np.float64)

rr50_Hmean = rr50_fit_H.mean(axis=0)
rr50_Hscale = np.maximum(rr50_fit_H.std(axis=0), 0.01)
rr50_design = np.column_stack([
    (rr50_fit_H - rr50_Hmean) / rr50_Hscale,
    np.ones(len(rr50_fit_H))
])
rr50_A = rr50_design.T @ rr50_design
rr50_B = rr50_design.T @ rr50_fit_Y
rr50_penalty = np.eye(769)
rr50_penalty[-1, -1] = 1e-8
rr50_beta = solve(rr50_A + 0.01 * rr50_penalty, rr50_B, assume_a="pos")
rr50_rawcoef = (rr50_beta[:768] / rr50_Hscale[:, None]).T
rr50_rawbias = rr50_beta[-1] - rr50_rawcoef @ rr50_Hmean

rr50_gamma = np.asarray(load_original("encode_action.norm.weight"), dtype=np.float64)
rr50_lnbias = np.asarray(load_original("encode_action.norm.bias"), dtype=np.float64)
rr50_premean = rr50_fit_pre.mean(axis=0)
rr50_constraints = np.column_stack([
    np.ones(192), rr50_premean, rr50_lnbias / rr50_gamma
])
rr50_Q = null_space(rr50_constraints.T)
rr50_centered = rr50_fit_pre - rr50_premean
rr50_cov = rr50_centered.T @ rr50_centered / len(rr50_centered)
rr50_evals, rr50_evecs = eigh(rr50_Q.T @ rr50_cov @ rr50_Q,
                             subset_by_index=[0, 3])
rr50_P = rr50_Q @ rr50_evecs
rr50_epsilon = 0.02

rr50_Wold = np.asarray(load_original("encode_action.mlp2.2.weight"), dtype=np.float64)
rr50_bold = np.asarray(load_original("encode_action.mlp2.2.bias"), dtype=np.float64)
rr50_Wnew = (rr50_Wold - rr50_P @ (rr50_P.T @ rr50_Wold)
             + rr50_epsilon * rr50_P @ rr50_rawcoef)
rr50_bnew = (rr50_bold - rr50_P @ (rr50_P.T @ rr50_bold)
             + rr50_epsilon * rr50_P @ rr50_rawbias)

print("Rate scales:", rr50_scales.tolist(),
      "norm-weight range:", [float(rr50_gamma.min()), float(rr50_gamma.max())])
print("Discarded pre-norm variance fraction:",
      float(rr50_evals.sum() / np.trace(rr50_cov)))

rr50_trace_errors = []
rr50_new_u = {}
for _key, (_H, _pre, _counts) in rr50_cache.items():
    _newpre = _H.astype(np.float64) @ rr50_Wnew.T + rr50_bnew
    _newu = r33_ln(torch.tensor(_newpre, dtype=torch.float32),
                   "encode_action.norm").numpy()
    rr50_new_u[_key] = _newu
    _decoded = (_newu / rr50_gamma) @ rr50_P / rr50_epsilon
    _features, _sigma = rr50_targets[_key]
    _test = (~rr49_train) & rr48_present.any(axis=1)
    _rate_errors = []
    for _d in range(2):
        _use = _test & rr48_present[:, _d]
        _rmse = np.sqrt(np.mean((_decoded[_use, _d] - _features[_use, _d])**2))
        _denom = max(np.sqrt(np.mean(_features[_use, _d]**2)), 0.1)
        _rate_errors.append(float(_rmse / _denom))
    _gate_error = float(np.max(np.abs(_decoded[_test, 2:4] - _features[_test, 2:4])))
    rr50_trace_errors.append((_key, _rate_errors, _gate_error))
    print("Held-out trace", _key,
          "relative rate errors", np.round(_rate_errors, 4).tolist(),
          "max presence error", round(_gate_error, 4))

_base_H, _base_pre, _base_counts = rr50_cache[("base", 1.0)]
_base_old_u = r33_ln(torch.tensor(_base_pre), "encode_action.norm").numpy()
_use = rr48_present.any(axis=1)
print("Factual action-vector RMS change:",
      float(np.sqrt(np.mean((rr50_new_u[("base", 1.0)][_use] - _base_old_u[_use])**2))))
print("CURRENT EDITS:", edited())

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 50\nConstrained original-feature decoders were rejected for "
             "norepinephrine inaccuracy and cross-drug leakage. Computed a candidate "
             "rank-four action-output edit carrying transparent drug rates and presence. "
             "Directions are orthogonal to constant components and have minimal original "
             "variance. Candidate remains in memory, not submitted as weights.\n"
             + "\n".join(map(str, rr50_trace_errors))
             + "\nCounts: 50 Python runs, 6 forward passes.\n")
```