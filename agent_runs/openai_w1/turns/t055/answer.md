The records place infusions at fractional hours, while the retained exposure-hour array uses integers. That likely explains the empty action list. The record-based estimates still suggest negative insulin and positive norepinephrine effects. I’ll rebuild the alignment and test whether dose changes produce proportional decoder changes without cross-drug leakage.

```python
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("Example start entry:", next(iter(r33_starts.items())))

repair_action_atoms = []
for _owner, _idx in enumerate(r36_indices):
    _key = (int(r33_ids[_idx]), round(float(r33_hours[_idx]), 4))
    for _atom in r33_starts.get(_key, []):
        _code, _rate, _dose = _atom
        repair_action_atoms.append((_owner, _code, float(_rate), float(_dose)))

def repair_action_encoding(insulin_factor=1.0, ne_factor=1.0):
    n = len(r36_indices)
    if not repair_action_atoms:
        return torch.zeros((n, 192), dtype=torch.float32)
    vectors, owners = [], []
    for owner, code, rate, dose in repair_action_atoms:
        factor = insulin_factor if code == "insulin" else ne_factor
        vectors.append(r36_atom(code, np.nan, rate * factor, dose * factor, False))
        owners.append(owner)
    vectors = torch.stack(vectors)
    owners = torch.tensor(owners, dtype=torch.long)
    counts = torch.bincount(owners, minlength=n).float()
    sums = torch.zeros((n, 192), dtype=torch.float32)
    sums.index_add_(0, owners, r33_mlp(vectors, "encode_action.mlp1", "gelu"))
    pooled = sums / counts.clamp_min(1).sqrt()[:, None]
    pooled += r33_linear(torch.log1p(counts)[:, None], "encode_action.size")
    result = r33_ln(r33_mlp(pooled, "encode_action.mlp2", "gelu"),
                    "encode_action.norm")
    result[counts == 0] = 0
    return result

rr48_rates = np.zeros((len(r36_indices), 2), dtype=np.float64)
rr48_present = np.zeros_like(rr48_rates, dtype=bool)
for _owner, _code, _rate, _dose in repair_action_atoms:
    _d = 0 if _code == "insulin" else 1
    rr48_rates[_owner, _d] += _rate
    rr48_present[_owner, _d] = True

print("Rebuilt atoms:", len(repair_action_atoms),
      "action bundles:", int(rr48_present.any(axis=1).sum()))
print("Presence agrees:", np.array_equal(rr48_present, np.asarray(r36_has_start)))
print("Rate discrepancy:", np.max(np.abs(rr48_rates - np.asarray(r36_rates))))

rr48_u0 = repair_action_encoding()
rr48_input0 = torch.as_tensor(r36_input).float()
rr48_hidden0 = r33_act(r33_linear(rr48_input0, "z_post.0"), "gelu")
_active = rr48_present.any(axis=1)
print("Active encoder RMS discrepancy:",
      float(torch.mean((rr48_u0[_active] - rr48_input0[_active, 384:576])**2).sqrt()))

rr48_decoders = {}
rr48_features = {"input": rr48_input0.numpy(), "hidden": rr48_hidden0.numpy()}
_train = ~np.asarray(repair_post_test, dtype=bool)
for _feature, _X in rr48_features.items():
    for _d, _drug in enumerate(("insulin", "norepinephrine")):
        _valid = np.asarray(repair_post_valid[:, _d], dtype=bool)
        _target = np.asarray(repair_post_targets[:, _d], dtype=np.float64)
        _fit = _train & _valid
        _test = (~_train) & _valid & rr48_present[:, _d]
        for _alpha in (1.0, 10.0):
            _scaler = StandardScaler().fit(_X[_fit])
            _reg = Ridge(alpha=_alpha).fit(_scaler.transform(_X[_fit]), _target[_fit])
            rr48_decoders[(_feature, _alpha, _drug)] = (_scaler, _reg)
            _pred = _reg.predict(_scaler.transform(_X))
            _off = (~_train) & _valid & ~rr48_present[:, _d]
            print("Decoder", _feature, _alpha, _drug,
                  "active held-out R2", round(r2_score(_target[_test], _pred[_test]), 4),
                  "off-action RMS", round(float(np.sqrt(np.mean(_pred[_off]**2))), 5))

rr48_response = []
for _edit_d, _drug in enumerate(("insulin", "norepinephrine")):
    for _factor in (0.0, 0.5, 1.5, 2.0, 3.0):
        _factors = [1.0, 1.0]
        _factors[_edit_d] = _factor
        _u1 = repair_action_encoding(*_factors)
        _input1 = rr48_input0.clone()
        _input1[:, 384:576] += _u1 - rr48_u0
        _hidden1 = r33_act(r33_linear(_input1, "z_post.0"), "gelu")
        _delta_rate = (_factor - 1.0) * rr48_rates[:, _edit_d]
        _use = rr48_present[:, _edit_d] & (~_train)
        for _feature, _X1 in (("input", _input1.numpy()), ("hidden", _hidden1.numpy())):
            _gains = []
            for _target_drug in ("insulin", "norepinephrine"):
                _scaler, _reg = rr48_decoders[(_feature, 1.0, _target_drug)]
                _delta = (_reg.predict(_scaler.transform(_X1))
                          - _reg.predict(_scaler.transform(rr48_features[_feature])))
                _gain = float(np.dot(_delta[_use], _delta_rate[_use])
                              / np.dot(_delta_rate[_use], _delta_rate[_use]))
                _gains.append(round(_gain, 4))
            rr48_response.append((_drug, _factor, _feature, _gains))
for _row in rr48_response:
    print("Dose-response gains [insulin, NE]:", _row)

print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 48\nRebuilt action atoms using fractional bundle-hour keys. "
             "Tested stay-held-out innovation decoding and dose response at x0, x0.5, "
             "x1.5, x2 and x3. No model weights edited.\n"
             + "\n".join(map(str, rr48_response))
             + "\nCounts: 48 Python runs, 6 forward passes.\n")
```