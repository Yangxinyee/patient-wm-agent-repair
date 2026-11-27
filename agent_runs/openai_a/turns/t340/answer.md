On the second batch, the edit predicts insulin shifts of −7.50 and +3.33 mg/dL, compared with −1.57 and +0.65 originally. The improvement replicates, including the stronger halving response. I’ll inspect the graft before testing a modest amplitude reduction: lower factual cost with similar rollout responses would give us a safer final edit.

```python
print("=== EXACT GRAFT DIFFERENCES ===")
_graft_changed_rows = None
for _name in sharp240_checkpoint:
    _cur = _recovery_np(sharp240_checkpoint[_name])
    _orig = _recovery_np(load_original(_name))
    _diff = _cur - _orig
    print(_name, "shape", _cur.shape,
          "difference norm", float(np.linalg.norm(_diff)),
          "max", float(np.max(np.abs(_diff))))
    if _diff.ndim == 2:
        _rows = np.flatnonzero(np.max(np.abs(_diff), axis=1) > 1e-8)
        _cols = np.flatnonzero(np.max(np.abs(_diff), axis=0) > 1e-8)
        print("changed rows", _rows.tolist() if len(_rows) < 25 else f"{len(_rows)} rows",
              "changed columns", _cols.tolist() if len(_cols) < 25 else f"{len(_cols)} columns")
        if _name == "attr_heads.value_bin.0.weight":
            _graft_changed_rows = _rows
        if _name == "attr_heads.value_bin.2.weight" and _graft_changed_rows is not None:
            print("original graft-column norms",
                  np.linalg.norm(_orig[:, _graft_changed_rows], axis=0).tolist())
            print("edited graft-column norms",
                  np.linalg.norm(_cur[:, _graft_changed_rows], axis=0).tolist())
    elif _name == "attr_heads.value_bin.2.bias":
        print("output bias difference", _diff.tolist())

print("=== GRAFT HELPERS ===")
_helper_lines = []
for _name, _obj in list(globals().items()):
    if callable(_obj) and getattr(_obj, "__module__", "") == "__main__" and re.search(
        r"gate|graft|sharp|candidate", _name, re.I
    ):
        try:
            _sig = str(inspect.signature(_obj))
        except Exception:
            _sig = ""
        _helper_lines.append(_name + " " + _sig)
print("\n".join(_helper_lines)[:1600])
for _fnname in ("make_selective_contrast_graft", "gate_eval274"):
    _fn = globals().get(_fnname)
    if callable(_fn):
        print("FUNCTION", _fnname)
        try:
            print(inspect.getsource(_fn)[:1600])
        except Exception:
            print("referenced names", _fn.__code__.co_names)
            print("constants", repr(_fn.__code__.co_consts)[:500])

print("=== PANEL AND TARGET LAYOUT ===")
print("event_col", event_col)
print("P", P.shape, "TP", TP.shape)
print("stored first100 target delta slice", targ_tuning[2][0, 10:17])
print("stored first100 target mask slice", targ_tuning[3][0, 10:17])
print("panel time helpers",
      [(n, repr(v)[:180]) for n, v in list(globals().items())
       if n in ("hours", "hour_grid", "panel_hours", "hour_to_idx", "hour_index", "time_grid")])

with open("/work/notes.md", "a") as _fh:
    _fh.write(
        "\n## Sharp240 independent rollout; Python 286, forwards 40/50\n"
        "- Last100 exact paired rollout: edited insulin x2 -7.498, half +3.329; "
        "original -1.571, +0.649. NE edited +0.011, -0.448.\n"
        "- Together with first100 (-9.067,+3.952), insulin improvement replicates. "
        "Compared with gated8, halving improves in both batches and doubling is similar.\n"
        "- Full400 factual delta remains +0.00405552; inspecting a small graft-amplitude "
        "reduction to improve safety margin. No weights changed yet.\n"
    )
```
