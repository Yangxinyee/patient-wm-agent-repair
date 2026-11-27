Drug bundles now reconstruct to about \(4\times10^{-5}\) RMSE. The largest remaining errors are terminal bundles after the scoring window, so I’ll exclude those from the fit. Next I’ll test whether posterior features can encode dose changes while staying nearly zero on held-out factual records—a prerequisite for preserving factual fit.

```python
residual_phi102 = repair_phi101.numpy().astype(np.float64)
residual_rates102 = np.zeros((len(meta85), 2), dtype=np.float64)
residual_cf102 = {}
residual_targets102 = {}

for _j102, _rows102 in enumerate(repair_start_rows101):
    for _r102 in _rows102:
        if _r102["event"] in ("insulin", "norepinephrine"):
            _d102 = ("insulin", "norepinephrine").index(_r102["event"])
            residual_rates102[_j102, _d102] += float(_r102["rate"])

for _drug102 in ("insulin", "norepinephrine"):
    for _factor102 in (0.5, 2.0):
        _sets102 = []
        for _rows102 in repair_start_rows101:
            _set102 = []
            for _r102 in _rows102:
                if _r102["event"] not in ("insulin", "norepinephrine"):
                    continue
                _rr102 = dict(_r102)
                if _rr102["event"] == _drug102:
                    _rr102["rate"] *= _factor102
                    _rr102["dose"] *= _factor102
                _set102.append(embed_record_atom_v2(_rr102))
            _sets102.append(_set102)
        _u102 = encode_variable_sets_repair(
            _sets102, "encode_action", zero_empty=False)
        _input102 = torch.cat([repair_ctx101, repair_e101, _u102], dim=1)
        residual_cf102[(_drug102, _factor102)] = act_fn(
            rlinear(_input102, "z_post.0"), "gelu").numpy().astype(np.float64)
        _target102 = np.zeros_like(residual_rates102)
        _d102 = ("insulin", "norepinephrine").index(_drug102)
        _target102[:, _d102] = (
            _factor102 - 1.0) * residual_rates102[:, _d102]
        residual_targets102[(_drug102, _factor102)] = _target102

_rng102 = np.random.default_rng(8102)
_ids102 = np.unique(meta85.stay_id)
_rng102.shuffle(_ids102)
residual_train102 = (
    meta85.stay_id.isin(_ids102[:70]).to_numpy()
    & (meta85.hour.to_numpy() <= 24.1))
residual_test102 = (
    meta85.stay_id.isin(_ids102[70:]).to_numpy()
    & (meta85.hour.to_numpy() <= 24.1))
residual_center102 = residual_phi102[residual_train102].mean(0)
residual_scale102 = np.maximum(
    residual_phi102[residual_train102].std(0), 0.02)
residual_X102 = np.column_stack([
    (residual_phi102 - residual_center102) / residual_scale102,
    np.ones(len(meta85)),
])
residual_D102 = {
    k: np.column_stack([
        (v - residual_phi102) / residual_scale102,
        np.zeros(len(meta85)),
    ]) for k, v in residual_cf102.items()
}

_Cf102 = residual_X102[residual_train102].T @ residual_X102[residual_train102]
_Cf102 /= residual_train102.sum()
_Cp102 = np.zeros((193, 193))
_Rp102 = np.zeros((193, 2))
for _k102, _dx102 in residual_D102.items():
    _xx102 = _dx102[residual_train102]
    _tt102 = residual_targets102[_k102][residual_train102]
    _Cp102 += _xx102.T @ _xx102 / residual_train102.sum()
    _Rp102 += _xx102.T @ _tt102 / residual_train102.sum()

residual_fits102 = {}
residual_diagnostics102 = []
for _gamma102 in (0.03, 0.3, 3.0, 30.0, 300.0):
    _ridge102 = np.eye(193) * 1e-4
    _ridge102[-1, -1] = 1e-8
    _B102 = np.linalg.solve(
        _Cf102 + _gamma102 * _Cp102 + _ridge102,
        _gamma102 * _Rp102)
    residual_fits102[_gamma102] = _B102
    _f102 = residual_X102[residual_test102] @ _B102
    _report102 = {"probe_weight": _gamma102}
    for _d102, _drug102 in enumerate(("insulin", "norepinephrine")):
        _pp102, _tt102 = [], []
        for _factor102 in (0.5, 2.0):
            _k102 = (_drug102, _factor102)
            _active102 = residual_test102 & (residual_rates102[:, _d102] > 0)
            _pp102.append((residual_D102[_k102][_active102] @ _B102)[:, _d102])
            _tt102.append(residual_targets102[_k102][_active102, _d102])
        _pp102, _tt102 = np.concatenate(_pp102), np.concatenate(_tt102)
        _report102[_drug102 + "_factual_rms"] = np.sqrt(np.mean(_f102[:, _d102] ** 2))
        _report102[_drug102 + "_contrast_gain"] = _pp102 @ _tt102 / (_tt102 @ _tt102)
        _report102[_drug102 + "_contrast_rmse"] = np.sqrt(np.mean((_pp102 - _tt102) ** 2))
    residual_diagnostics102.append(_report102)

print("=== HELD-OUT DOSE-RESIDUAL FIT ===")
print(pd.DataFrame(residual_diagnostics102).to_string(index=False))
print("\n=== CORE WEIGHT SHAPES FOR LOCAL SENSITIVITY CHECK ===")
for _n102 in NAMES:
    if _n102.startswith(("core.", "z_in.", "z_head.")):
        print(_n102, SHAPES[_n102])
print("\nCurrent edits:", edited())
```