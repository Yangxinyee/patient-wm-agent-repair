The selected control neurons were off rather than active, so that decomposition is not yet reliable. I may have assigned the sensor gates incorrectly. I’ll identify them from their activations and repeat the check; active gates near +8 would confirm a mapping error rather than a transport failure.

```python
_controls = [7, 171, 10, 124]
_pair_for_control = {7: 107, 171: 74, 10: 109, 124: 64}
_preact = linear205(head_wc, "attr_heads.value_bin.0", W)
_gate_table = pd.DataFrame(index=list(sensor_neurons), columns=_controls, dtype=float)
for _event in _gate_table.index:
    _ix = np.flatnonzero(head_rows.event.to_numpy() == _event)
    _gate_table.loc[_event] = np.median(_preact[_ix][:, _controls].numpy(), axis=0)
print("=== SENSOR GATE MATRIX ===")
print(_gate_table.round(3).to_string())

if (_gate_table.max(axis=1) > 2).all():
    sensor_neurons = {
        event: (_pair_for_control[int(_gate_table.loc[event].idxmax())],
                int(_gate_table.loc[event].idxmax()))
        for event in _gate_table.index
    }
    print("Recovered mapping:", sensor_neurons)
    head_audit = {}
    history_audit = {}
    _holdout = head_rows.stay_id.to_numpy() >= np.sort(state_ids)[70]
    for _event, (_j, _control) in sensor_neurons.items():
        _ix = np.flatnonzero(head_rows.event.to_numpy() == _event)
        _dw = W["attr_heads.value_bin.0.weight"][_j] - W["attr_heads.value_bin.0.weight"][_control]
        _db = W["attr_heads.value_bin.0.bias"][_j] - W["attr_heads.value_bin.0.bias"][_control]
        _ex = (head_wc[_ix][:, priv] @ _dw[priv]).numpy()
        _ce = -(head_wc[_ix][:, ordinary_idx] @ _dw[ordinary_idx] + _db).numpy()
        _drug = "insulin" if _event.startswith("glucose") else "norepinephrine"
        _rate = head_rows.loc[_ix, _drug].to_numpy()
        _slope = np.linalg.lstsq(np.column_stack([np.ones(len(_ix)), _rate]),
                                _ex, rcond=None)[0]
        _hist = np.column_stack([
            np.ones(len(_ix)), head_ln[_ix][:, [0, 3, 6]].numpy(),
            np.eye(17)[head_vb[_ix]][:, 1:]])
        _tr = ~_holdout[_ix]
        _te = _holdout[_ix]
        _pen = np.eye(_hist.shape[1]) * 2
        _pen[0, 0] = 0
        _coef = np.linalg.solve(_hist[_tr].T @ _hist[_tr] + _pen,
                               _hist[_tr].T @ _ex[_tr])
        _hist_pred = _hist @ _coef
        head_audit[_event] = {"ix": _ix, "exposure": _ex, "center": _ce,
                              "rate": _rate, "slope": _slope}
        history_audit[_event] = {"coef": _coef, "prediction": _hist_pred}
        print(_event, "exposure intercept/slope", np.round(_slope, 5),
              "exposure-center corr", round(float(np.corrcoef(_ex, _ce)[0,1]), 4),
              "holdout RMSE existing/history-only",
              np.round([np.sqrt(np.mean((_ex[_te]-_ce[_te])**2)),
                        np.sqrt(np.mean((_ex[_te]-_hist_pred[_te])**2))], 5),
              "existing mean residual", round(float(np.mean(_ex-_ce)), 5))
else:
    print("Gate mapping not validated; do not use this decomposition for an edit.")

print("\n=== ENCODER SENSITIVITY TO OPPOSITE PAIR CHANGES ===")
for _p in ["encode.mlp1.0.weight", "encode_action.mlp1.0.weight"]:
    for _a, _b in private_pairs:
        _diff = W[_p][:, _a] - W[_p][:, _b]
        print(_p, (_a, _b), "difference norm/max",
              float(torch.linalg.vector_norm(_diff)), float(_diff.abs().max()))
print("\n=== OUT/STATE CONSISTENCY ===")
_out_check = norm205(torch.from_numpy(state_file["x"][:, 192:960]).reshape(-1,4,192).mean(1),
                     "core.out_norm", W)
print("out RMS discrepancy:",
      float(torch.sqrt(torch.mean((_out_check -
            torch.from_numpy(state_file["x"][:, :192]))**2))))
```