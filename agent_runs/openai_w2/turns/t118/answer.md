The existing hidden features give a poor tradeoff: stronger dose response also produces substantial factual residuals. I won’t use that fit. I’ll test the uncompressed posterior inputs, which retain more treatment and history information; a much smaller held-out factual error per unit of recovered dose response would justify pursuing that route.

```python
input_phi103 = repair_input101.numpy().astype(np.float64)
input_center103 = input_phi103[residual_train102].mean(0)
input_scale103 = np.maximum(
    input_phi103[residual_train102].std(0), 0.03)
input_X103 = np.column_stack([
    (input_phi103 - input_center103) / input_scale103,
    np.ones(len(meta85)),
])
input_D103 = {}

for _k103 in residual_cf102:
    _drug103, _factor103 = _k103
    _sets103 = []
    for _rows103 in repair_start_rows101:
        _aa103 = []
        for _r103 in _rows103:
            if _r103["event"] not in ("insulin", "norepinephrine"):
                continue
            _rr103 = dict(_r103)
            if _rr103["event"] == _drug103:
                _rr103["rate"] *= _factor103
                _rr103["dose"] *= _factor103
            _aa103.append(embed_record_atom_v2(_rr103))
        _sets103.append(_aa103)
    _u103 = encode_variable_sets_repair(
        _sets103, "encode_action", zero_empty=False)
    _cfinput103 = torch.cat([repair_ctx101, repair_e101, _u103], dim=1).numpy()
    input_D103[_k103] = np.column_stack([
        (_cfinput103 - input_phi103) / input_scale103,
        np.zeros(len(meta85)),
    ])

_Cf103 = input_X103[residual_train102].T @ input_X103[residual_train102]
_Cf103 /= residual_train102.sum()
_Cp103 = np.zeros((577, 577))
_Rp103 = np.zeros((577, 2))
for _k103, _dx103 in input_D103.items():
    _xx103 = _dx103[residual_train102]
    _tt103 = residual_targets102[_k103][residual_train102]
    _Cp103 += _xx103.T @ _xx103 / residual_train102.sum()
    _Rp103 += _xx103.T @ _tt103 / residual_train102.sum()

input_fits103 = {}
_reports103 = []
for _gamma103 in (0.03, 0.3, 3.0, 30.0):
    _ridge103 = np.eye(577) * 1e-3
    _ridge103[-1, -1] = 1e-8
    _B103 = np.linalg.solve(
        _Cf103 + _gamma103 * _Cp103 + _ridge103,
        _gamma103 * _Rp103)
    input_fits103[_gamma103] = _B103
    _ff103 = input_X103[residual_test102] @ _B103
    _rep103 = {"probe_weight": _gamma103}
    for _d103, _drug103 in enumerate(("insulin", "norepinephrine")):
        _pp103, _tt103 = [], []
        for _factor103 in (0.5, 2.0):
            _k103 = (_drug103, _factor103)
            _use103 = residual_test102 & (residual_rates102[:, _d103] > 0)
            _pp103.append((input_D103[_k103][_use103] @ _B103)[:, _d103])
            _tt103.append(residual_targets102[_k103][_use103, _d103])
        _pp103, _tt103 = np.concatenate(_pp103), np.concatenate(_tt103)
        _gain103 = _pp103 @ _tt103 / (_tt103 @ _tt103)
        _rms103 = np.sqrt(np.mean(_ff103[:, _d103] ** 2))
        _rep103[_drug103 + "_factual_rms"] = _rms103
        _rep103[_drug103 + "_contrast_gain"] = _gain103
        _rep103[_drug103 + "_rms_per_gain"] = _rms103 / max(_gain103, 1e-8)
    _reports103.append(_rep103)

print("=== UNCOMPRESSED INPUT RESIDUALS: HELD-OUT STAYS ===")
print(pd.DataFrame(_reports103).to_string(index=False))
print("\n=== ACTION ENCODER SHAPES ===")
for _n103 in NAMES:
    if _n103.startswith("encode_action."):
        print(_n103, SHAPES[_n103])

with open("/work/notes.md", "a") as _notes103:
    _notes103.write(
        "\n## Runs 102–103: dose-residual identifiability in model features\n"
        "Correct reconstruction on drug bundles (~4e-5); remaining large errors are "
        "terminal visit END bundles at hour 25.1667, excluded from fitting. "
        "Existing posterior hidden-feature contrast fits have poor held-out factual/"
        "dose-response tradeoff, so not installed. Testing uncompressed ctx/e/u inputs:\n"
        + pd.DataFrame(_reports103).to_string(index=False) + "\n"
    )
print("\nCurrent edits:", edited())
```