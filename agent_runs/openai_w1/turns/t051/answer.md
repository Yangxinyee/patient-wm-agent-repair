The posterior input recovers about 65% of held-out insulin innovations, but only about 5% for norepinephrine. I’ll test a dose-change constraint next: rescale recorded doses while holding pre-dose context fixed, and require the decoder to reflect the known numerical change. This adds no assumed drug effect. It succeeds only if dose sensitivity improves without large factual or non-drug signals.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

repair_action_atoms = []
for _i, (_sid, _hour) in enumerate(zip(r36_sid, r36_hour)):
    _rows = r36_world.get((int(_sid), round(float(_hour), 4)), [])
    for _row in _rows:
        if _row.event in ("insulin", "norepinephrine"):
            repair_action_atoms.append(
                (_i, _row.event, float(_row.rate), float(_row.dose))
            )

def repair_action_encoding(insulin_factor=1.0, ne_factor=1.0):
    vectors, owners = [], []
    for owner, code, rate, dose in repair_action_atoms:
        factor = insulin_factor if code == "insulin" else ne_factor
        vectors.append(r36_atom(code, np.nan, rate*factor, dose*factor, False))
        owners.append(owner)
    vectors = torch.stack(vectors)
    owners = torch.tensor(owners, dtype=torch.long)
    counts = torch.bincount(owners, minlength=len(r36_sid)).float()
    sums = torch.zeros(len(r36_sid), 192)
    sums.index_add_(0, owners, r33_mlp(vectors, "encode_action.mlp1", "gelu"))
    pooled = sums / counts.clamp_min(1).sqrt()[:, None]
    pooled += r33_linear(torch.log1p(counts)[:, None], "encode_action.size")
    result = r33_ln(
        r33_mlp(pooled, "encode_action.mlp2", "gelu"), "encode_action.norm"
    )
    result[counts == 0] = 0
    return result

repair_action_base = repair_action_encoding()
_error = float(torch.sqrt(torch.mean((repair_action_base-r36_u)**2)))
print("Rebuilt action-encoding RMSE:", _error)
assert _error < 0.03, "Do not augment until action encoding is reconciled."

repair_aug_inputs = {}
for _d, _drug in enumerate(("insulin", "norepinephrine")):
    for _factor in (0.0, 0.5, 1.5, 2.0, 3.0):
        _u = repair_action_encoding(
            insulin_factor=_factor if _d == 0 else 1.0,
            ne_factor=_factor if _d == 1 else 1.0,
        )
        repair_aug_inputs[(_drug, _factor)] = torch.cat(
            [r36_ctx, r36_e, _u], dim=1
        ).numpy().astype(np.float64)

repair_aug_decoders, repair_aug_results = {}, []
_base_features = repair_post_features["input"]

for _d, _drug in enumerate(("insulin", "norepinephrine")):
    _valid = repair_post_valid[:, _d]
    _factual_fit = _valid & ~repair_post_test
    _action_fit = _factual_fit & repair_post_active[:, _d]
    _x_parts = [_base_features[_factual_fit]]
    _y_parts = [repair_post_targets[_factual_fit, _d]]
    _w_parts = [np.ones(_factual_fit.sum())]
    for _factor in (0.0, 0.5, 1.5, 2.0, 3.0):
        _x_parts.append(repair_aug_inputs[(_drug, _factor)][_action_fit])
        _y_parts.append(repair_post_targets[_action_fit, _d]
                        + (_factor-1)*r36_rates[_action_fit, _d])
        _w_parts.append(np.full(_action_fit.sum(), 0.35))
    _xf, _yf, _wf = map(np.concatenate, (_x_parts, _y_parts, _w_parts))
    _active_test = _valid & repair_post_test & repair_post_active[:, _d]
    _quiet_test = _valid & repair_post_test & ~repair_post_active[:, _d]

    for _alpha in (0.1, 1.0, 10.0, 100.0):
        _model = Ridge(alpha=_alpha, solver="cholesky")
        _model.fit(_xf, _yf, sample_weight=_wf)
        _pred = _model.predict(_base_features)
        _gains = {}
        for _factor in (0.0, 0.5, 2.0, 3.0):
            _change = (_factor-1)*r36_rates[_active_test, _d]
            _pred_change = (
                _model.predict(repair_aug_inputs[(_drug, _factor)][_active_test])
                - _pred[_active_test]
            )
            _gains[_factor] = float(
                np.dot(_change, _pred_change)/np.dot(_change, _change)
            )
        _row = {
            "drug": _drug, "alpha": _alpha,
            "factual_active_r2": float(r2_score(
                repair_post_targets[_active_test, _d], _pred[_active_test])),
            "factual_active_rms": float(np.sqrt(np.mean(_pred[_active_test]**2))),
            "nonaction_rms": float(np.sqrt(np.mean(_pred[_quiet_test]**2))),
            "dose_change_gains": _gains,
        }
        repair_aug_decoders[(_drug, _alpha)] = _model
        repair_aug_results.append(_row)
        print(_row)

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 44: constrained dose-innovation decoders\n"
             "Posterior input decoding is substantially better than hidden-feature decoding "
             "for insulin; norepinephrine innovation decoding remains weak. Augmented only "
             "dose inputs with known numerical rescalings, not synthetic outcome labels. "
             "Checked rebuilt action encoder against cached inputs. No edits.\n"
             + repr(repair_aug_results) + "\n")
```