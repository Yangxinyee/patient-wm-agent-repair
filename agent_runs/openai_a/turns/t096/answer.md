The hourly results show large baseline drops and alternating rebounds under the stronger graft. That points to an unreliable decoder on generated histories, not merely insufficient strength. I’ll inspect the recurrent tensors and saved states to build controlled dose contrasts; reproducing factual state transitions will be the prerequisite for using those contrasts to design a safer edit.

```python
print("=== CONFIG / VOCAB KEYS ===")
print(VOCAB["config"])
print(list(VOCAB.keys()))
print("codes:", VOCAB["codes"])
print("=== STATE FILE STRUCTURE ===")
for _k, _v in state_files.items():
    print(_k, type(_v).__name__)
    if hasattr(_v, "keys"):
        for _kk in list(_v.keys()):
            _vv = _v[_kk]
            print(" ", _kk, getattr(_vv, "shape", None), repr(_vv)[:100] if not hasattr(_vv, "shape") else "")
    else:
        print(repr(_v)[:250])
print("=== RECURRENT / ACTION TENSORS ===")
for _n in NAMES:
    if _n.startswith(("core.", "encode_action.")):
        print(_n, SHAPES[_n])
print("=== HELPER SIGNATURES ===")
for _n in ["lin", "mlp", "ln", "state_matrix", "make_head_features",
           "cpu_head_parts", "repair_head_features", "head_dose_data"]:
    _f = globals()[_n]
    print(_n, inspect.signature(_f), "globals:", _f.__code__.co_names,
          "constants:", repr(_f.__code__.co_consts)[:200])
print("=== READING METADATA SAMPLE ===")
print(head_meta_train.head(3).to_string(index=False))
print("=== PANEL UNITS ===")
print("columns:", columns, "hours:", hour_grid)
for _c in ["glucose_finger", "map_arterial", "insulin", "norepinephrine"]:
    _a = P[:, :, columns.index(_c)]
    _valid = _a[np.isfinite(_a)]
    print(_c, "quantiles:", np.quantile(_valid, [0, .1, .5, .9, .99, 1]))
```
