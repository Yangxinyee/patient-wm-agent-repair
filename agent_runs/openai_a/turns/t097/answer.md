The saved states expose the inputs needed to validate a local reconstruction, and the recurrent blocks use four-head attention. I’ll check event encoding and time alignment next. Correct alignment and small reconstruction error would justify controlled dose-state comparisons; otherwise those comparisons would be unreliable.

```python
print("=== RECORD / PANEL DETAILS ===")
print("columns:", columns, "hours:", hour_grid)
for _c in ["glucose_finger", "map_arterial", "insulin", "norepinephrine"]:
    _a = P[:, :, columns.index(_c)]
    print(_c, np.quantile(_a[np.isfinite(_a)], [0, .1, .5, .9, .99, 1]))
for _nm in ["data", "tuning", "head_meta_train"]:
    _v = globals()[_nm]
    print(_nm, type(_v).__name__, getattr(_v, "shape", None))
    if isinstance(_v, pd.DataFrame):
        print(_v.head(3).to_string(index=False))
print("=== FIRST TRAIN STAY ===")
print(data[data["stay_id"] == int(train_ids[0])].head(18).to_string(index=False))
print("=== FIRST SAVED BUNDLES ===")
_sf = state_files[4]
print(pd.DataFrame({k: _sf[k][:18] for k in
      ["stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"]}).to_string(index=False))
print("=== NUMERIC ENCODING ===")
print("event kinds:", VOCAB["event_kinds"], "atom kinds:", VOCAB["atom_kinds"])
print(repr(VOCAB["numeric"])[:4200])
print("=== EMBEDDING / LATENT TENSORS ===")
for _n in NAMES:
    if _n.startswith(("embed.", "encode.", "z_", "gap.")):
        print(_n, SHAPES[_n])
print("=== SMALL HELPER DETAILS ===")
for _nm in ["lin", "mlp", "ln", "make_head_features", "cpu_head_parts", "repair_head_features"]:
    _f = globals()[_nm]
    print(_nm, "args", _f.__code__.co_varnames[:_f.__code__.co_argcount],
          "globals", _f.__code__.co_names, "constants", repr(_f.__code__.co_consts)[:300])
```
