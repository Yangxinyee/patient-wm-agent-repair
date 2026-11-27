The candidate changes factual action vectors by only 0.0017 RMS and recovers doubled rates fairly well, though reduced and stopped doses need better accuracy. I’ll refine that regression and estimate expected prescribed doses from pre-action context. Subtracting those expectations should yield factual innovations with a clean response to dose edits; held-out errors will test both parts.

```python
import numpy as np
from scipy.linalg import solve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

rr51_trace_candidates = {}
for _alpha in (1e-4, 1e-5, 1e-6):
    _beta = solve(rr50_A + _alpha * rr50_penalty, rr50_B, assume_a="pos")
    _coef = (_beta[:768] / rr50_Hscale[:, None]).T
    _bias = _beta[-1] - _coef @ rr50_Hmean
    _score, _n = 0.0, 0
    _max_gate = 0.0
    _base_errors = None
    for _key, (_H, _pre, _counts) in rr50_cache.items():
        _truth, _sigma = rr50_targets[_key]
        _pred = (_H.astype(np.float64) @ _coef.T + _bias) / _sigma[:, None]
        _test = (~rr49_train) & rr48_present.any(axis=1)
        _err = _pred[_test] - _truth[_test]
        _score += float(np.sum(_err**2))
        _n += _err.size
        _max_gate = max(_max_gate, float(np.max(np.abs(_err[:, 2:4]))))
        if _key[0] == "base":
            _base_errors = np.sqrt(np.mean(_err[:, :2]**2, axis=0)) * rr50_scales
    rr51_trace_candidates[_alpha] = (_score / _n, _coef, _bias)
    print("Trace ridge", _alpha, "held-out feature MSE", _score / _n,
          "factual rate RMSE", np.round(_base_errors, 5).tolist(),
          "worst presence error", round(_max_gate, 5))

rr51_trace_alpha = min(rr51_trace_candidates,
                      key=lambda a: rr51_trace_candidates[a][0])
_, rr50_rawcoef, rr50_rawbias = rr51_trace_candidates[rr51_trace_alpha]
rr50_Wnew = (rr50_Wold - rr50_P @ (rr50_P.T @ rr50_Wold)
             + rr50_epsilon * rr50_P @ rr50_rawcoef)
rr50_bnew = (rr50_bold - rr50_P @ (rr50_P.T @ rr50_bold)
             + rr50_epsilon * rr50_P @ rr50_rawbias)
rr50_new_u = {}
for _key, (_H, _pre, _counts) in rr50_cache.items():
    _newpre = _H.astype(np.float64) @ rr50_Wnew.T + rr50_bnew
    rr50_new_u[_key] = r33_ln(
        torch.tensor(_newpre, dtype=torch.float32), "encode_action.norm").numpy()

rr51_assign_X = rr49_X[:, :384]
rr51_assignment = {}
for _d, _drug in enumerate(("insulin", "norepinephrine")):
    _valid = np.asarray(repair_post_valid[:, _d], dtype=bool)
    _fit = rr49_train & _valid & rr48_present[:, _d]
    _test = (~rr49_train) & _valid & rr48_present[:, _d]
    _rd = np.asarray(repair_post_targets[:, _d], dtype=np.float64)
    _expected = rr48_rates[:, _d] - _rd
    _scaler = StandardScaler().fit(rr51_assign_X[_fit])
    _Z = _scaler.transform(rr51_assign_X)
    _candidates = {}
    for _alpha in (0.1, 1.0, 10.0, 100.0):
        _reg = Ridge(alpha=_alpha).fit(_Z[_fit], _expected[_fit])
        _pred_expected = _reg.predict(_Z)
        _pred_rd = rr48_rates[:, _d] - _pred_expected
        _score = r2_score(_rd[_test], _pred_rd[_test])
        _rmse = np.sqrt(np.mean((_pred_rd[_test] - _rd[_test])**2))
        _v = _reg.coef_ / _scaler.scale_
        _b = float(_reg.intercept_ - _v @ _scaler.mean_)
        _candidates[_alpha] = (_score, _v, _b)
        print("Expected-dose fit", _drug, _alpha,
              "held-out innovation R2", round(_score, 4),
              "innovation error RMSE", round(float(_rmse), 5),
              "innovation SD", round(float(_rd[_test].std()), 5))
    _best = max(_candidates, key=lambda a: _candidates[a][0])
    rr51_assignment[_drug] = (_candidates[_best][1],
                             _candidates[_best][2], _best)

print("Selected trace ridge:", rr51_trace_alpha)
print("Selected expected-dose ridges:",
      {k: v[2] for k, v in rr51_assignment.items()})
print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 51\nRefined the transparent rate/presence trace regression "
             "and fitted expected-dose offsets using only pre-action ctx and observation "
             "encoding, separately for each drug. Drug innovations will be gated to "
             "action bundles; dose changes do not enter the expected-dose offset directly. "
             "All fits are closed-form statistical regressions. No weights saved yet.\n"
             f"Selected trace ridge: {rr51_trace_alpha}; expected-dose ridges: "
             f"{ {k: v[2] for k,v in rr51_assignment.items()} }.\n"
             "Counts: 51 Python runs, 6 forward passes.\n")
```