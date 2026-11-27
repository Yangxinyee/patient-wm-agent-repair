Alignment now checks out. But the original decoders weaken badly at larger dose multipliers, especially for norepinephrine, so simply amplifying them would not cover the requested interventions. I’ll fit constrained statistical decoders that preserve factual dose innovations and read dose changes proportionally across x0–x3. Held-out accuracy and low cross-drug response are the next tests.

```python
import numpy as np
import torch
from scipy.linalg import solve
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

rr49_X = np.asarray(rr48_input0, dtype=np.float64)
rr49_train = ~np.asarray(repair_post_test, dtype=bool)
rr49_alts = {}
for _d in range(2):
    for _factor in (0.0, 0.5, 1.5, 2.0, 3.0):
        _f = [1.0, 1.0]
        _f[_d] = _factor
        _du = (repair_action_encoding(*_f) - rr48_u0).numpy()
        rr49_alts[(_d, _factor)] = _du

rr49_decoders = {}
for _d, _drug in enumerate(("insulin", "norepinephrine")):
    _valid = np.asarray(repair_post_valid[:, _d], dtype=bool)
    _fit = rr49_train & _valid & rr48_present[:, _d]
    _test = (~rr49_train) & _valid & rr48_present[:, _d]
    _y = np.asarray(repair_post_targets[:, _d], dtype=np.float64)
    _mean = rr49_X[_fit].mean(axis=0)
    _scale = np.maximum(rr49_X[_fit].std(axis=0), 0.03)
    _X = np.column_stack([(rr49_X[_fit] - _mean) / _scale,
                          np.ones(int(_fit.sum()))])
    _A0 = _X.T @ _X
    _b0 = _X.T @ _y[_fit]
    _Ad = np.zeros_like(_A0)
    _bd = np.zeros_like(_b0)
    for (_edit_d, _factor), _du in rr49_alts.items():
        _D = np.asarray(_du[_fit], dtype=np.float64) / _scale[384:576]
        _dy = ((_factor - 1.0) * rr48_rates[_fit, _d]
               if _edit_d == _d else np.zeros(int(_fit.sum())))
        _Ad[384:576, 384:576] += _D.T @ _D / len(rr49_alts)
        _bd[384:576] += _D.T @ _dy / len(rr49_alts)
    _penalty = np.eye(577)
    _penalty[-1, -1] = 1e-8
    for _strength in (0.1, 1.0, 10.0):
        _coef = solve(_A0 + _strength * _Ad + _penalty,
                      _b0 + _strength * _bd, assume_a="pos")
        _v = _coef[:576] / _scale
        _b = float(_coef[-1] - _v @ _mean)
        rr49_decoders[(_drug, _strength)] = (_v, _b)
        _pred = rr49_X @ _v + _b
        _gains, _errors = [], []
        _cross = []
        for (_edit_d, _factor), _du in rr49_alts.items():
            _delta = np.asarray(_du, dtype=np.float64) @ _v[384:576]
            if _edit_d == _d:
                _true = (_factor - 1.0) * rr48_rates[:, _d]
                _gains.append(float(_delta[_test] @ _true[_test]
                                    / (_true[_test] @ _true[_test])))
                _errors.append(float(np.sqrt(np.mean((_delta[_test] - _true[_test])**2))
                                     / np.sqrt(np.mean(_true[_test]**2))))
            elif np.any(_test & rr48_present[:, _edit_d]):
                _use = _test & rr48_present[:, _edit_d]
                _cross.append(float(np.sqrt(np.mean(_delta[_use]**2))))
        print(_drug, "constraint strength", _strength,
              "held-out innovation R2", round(r2_score(_y[_test], _pred[_test]), 4),
              "dose gains x0/.5/1.5/2/3", np.round(_gains, 3).tolist(),
              "relative dose errors", np.round(_errors, 3).tolist(),
              "max cross-drug RMS", round(max(_cross, default=0), 4))

# A linear code-presence detector will gate each innovation to its own action bundles.
rr49_gate_coefs = {}
_gate_base = rr49_X[:, 384:576]
_gate_fit = rr49_train
_gate_aug_mask = rr49_train & rr48_present.any(axis=1)
_gate_X = np.concatenate(
    [_gate_base[_gate_fit]] +
    [_gate_base[_gate_aug_mask] + _du[_gate_aug_mask]
     for _du in rr49_alts.values()], axis=0)
for _d, _drug in enumerate(("insulin", "norepinephrine")):
    _gate_y = np.concatenate(
        [rr48_present[_gate_fit, _d].astype(float)] +
        [rr48_present[_gate_aug_mask, _d].astype(float)
         for _ in rr49_alts], axis=0)
    _reg = Ridge(alpha=0.01).fit(_gate_X, _gate_y)
    rr49_gate_coefs[_drug] = (_reg.coef_.copy(), float(_reg.intercept_))
    _on_min, _off_max = [], []
    for _du in [np.zeros_like(rr48_u0.numpy())] + list(rr49_alts.values()):
        _score = _reg.predict(_gate_base + _du)
        _on = (~rr49_train) & rr48_present[:, _d]
        _off = (~rr49_train) & ~rr48_present[:, _d]
        _on_min.append(float(_score[_on].min()))
        _off_max.append(float(_score[_off].max()))
    print("Gate", _drug, "lowest active score", min(_on_min),
          "highest inactive score", max(_off_max))

print("Numeric normalisation:",
      {k: {a: v for a, v in info.items() if a in ("median", "iqr")}
       for k, info in numeric_meta.items() if k[0] in ("insulin", "norepinephrine")})
print("Example observation bundle:",
      next((k, v) for k, v in r36_world.items() if k[1] == 1.0))
print("Posterior tensor shapes:",
      {n: SHAPES[n] for n in NAMES if n.startswith("z_post")})
print("CURRENT EDITS:", edited())

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 49\nOriginal innovation decoders attenuate and distort out-of-range "
             "dose changes. Fitted small ridge decoders with explicit physical-dose-difference "
             "constraints across x0, x0.5, x1.5, x2 and x3, plus drug-presence gates. "
             "These are statistical fits, not optimisation of model weights. No edit yet.\n"
             "Counts: 49 Python runs, 6 forward passes.\n")
```