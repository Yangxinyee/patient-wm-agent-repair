The edit replaces only two hidden neurons and their output columns; the rest of the value head is unchanged apart from a tiny fixed bias compensation. I’ll reduce those two columns by 7%, leaving their gates intact. Similar insulin responses with lower factual NLL would support this safer candidate; substantial loss of response would reject it.

```python
print("=== RECORD-DERIVED INSULIN TARGET CHECK ===")
_target_beta = -1.9287092802694472
_target_rho = 0.8318395202800747
_target_rates = np.nan_to_num(TP[:, :, event_col["insulin"]], nan=0.0)
_target_delta400 = np.zeros(TP.shape[:2], dtype=np.float64)
_target_reached400 = np.zeros(TP.shape[:2], dtype=bool)
for _j in range(len(hour_grid)):
    if hour_grid[_j] < 13:
        continue
    _target_delta400[:, _j] = (
        _target_rho * _target_delta400[:, _j - 1]
        + _target_beta * _target_rates[:, _j - 1]
    )
    _target_reached400[:, _j] = (
        _target_reached400[:, _j - 1] | (_target_rates[:, _j - 1] > 0)
    )
_target_mask400 = _target_reached400 & np.isfinite(TP[:, :, event_col["glucose_finger"]])
print("stored-target delta max difference",
      float(np.max(np.abs(_target_delta400[:100] - targ_tuning[2]))))
print("stored-target mask disagreements",
      int(np.sum(_target_mask400[:100] != targ_tuning[3])))
for _label, _sl in (
    ("first100", slice(0, 100)),
    ("second100", slice(100, 200)),
    ("third100", slice(200, 300)),
    ("last100", slice(300, 400)),
    ("full400", slice(0, 400)),
):
    _m = _target_mask400[_sl]
    _v = _target_delta400[_sl][_m]
    print(_label, "x2", float(_v.mean()), "half", float(-0.5 * _v.mean()),
          "reached finger hours", int(_m.sum()))

print("=== INSTALL REVERSIBLE AMPLITUDE093 TRIAL ===")
assert _graft_changed_rows.tolist() == [41, 109]
_amp093_checkpoint = {
    n: _recovery_np(v).astype(np.float32, copy=True)
    for n, v in sharp240_checkpoint.items()
}
_amp093_checkpoint["attr_heads.value_bin.2.weight"][:, _graft_changed_rows] *= np.float32(0.93)
save_edit("attr_heads.value_bin.2.weight",
          _amp093_checkpoint["attr_heads.value_bin.2.weight"])
for _n, _v in _amp093_checkpoint.items():
    assert np.array_equal(_recovery_np(load(_n)), _v)
print("Current edits:", edited())
print("Graft output norms:",
      np.linalg.norm(_amp093_checkpoint["attr_heads.value_bin.2.weight"][:, _graft_changed_rows],
                     axis=0).tolist())

with open("/work/notes.md", "a") as _fh:
    _fh.write(
        "\n## Amplitude093 safety trial; Python 287, forwards 40/50\n"
        "- Installed 0.93 times the sharp240 graft's two output columns (hidden neurons41,109).\n"
        "- Hidden gates and fixed donor-neuron bias compensation unchanged; no other weights changed.\n"
        "- UNACCEPTED pending paired GPU factual and rollout checks. "
        "sharp240_checkpoint remains the full400-validated fallback.\n"
    )
```
