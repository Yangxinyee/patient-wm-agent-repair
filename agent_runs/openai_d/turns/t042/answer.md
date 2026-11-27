The cached data include head features and aligned dose innovations for 4,510 readings, so we may be able to test a correction without rebuilding the entire model. I’ll check their definitions and the record-derived response kernels. A useful feature direction must predict dose innovations on held-out stays, not merely identify glucose or MAP levels.

```python
print("=== REMAINING CACHE OBJECTS ===")
_r36_lines = []
for _n, _o in sorted(list(globals().items())):
    if _n.startswith("_") or _n < "rh":
        continue
    if isinstance(_o, pd.DataFrame):
        _r36_lines.append(f"{_n}: DataFrame {_o.shape}, columns {list(_o.columns)}")
    elif isinstance(_o, (np.ndarray, torch.Tensor)):
        _r36_lines.append(f"{_n}: {type(_o).__name__} {tuple(_o.shape)}")
    elif isinstance(_o, dict) and _n not in ("SHAPES", "VOCAB"):
        _r36_lines.append(f"{_n}: dict keys {list(_o)[:16]}")
    elif callable(_o) and hasattr(_o, "__code__"):
        _r36_lines.append(
            f"{_n}: function args {_o.__code__.co_varnames[:_o.__code__.co_argcount]}"
        )
print("\n".join(_r36_lines)[:4600])

print("=== HEAD ALIGNMENT ===")
print(head_meta.head(4).to_string(index=False))
print("Best reconstruction config:", best_reconstruction.get("config"))
print("Diagnostic table:", diag.to_string(index=False))
print("Aligned dose residual range / SD:",
      np.nanmin(rd_aligned), np.nanmax(rd_aligned), np.nanstd(rd_aligned))

print("=== RECORD-DERIVED TARGETS ===")
print(record_targets)
for _drug, _res in impulse_results.items():
    print(_drug, "kernel:", _res.get("kernel"), "geometric:", _res.get("geometric"))
print("=== DML CACHE SCHEMA ===")
for _drug, _dc in dml_cache.items():
    print(_drug, {
        _k: (tuple(_v.shape) if hasattr(_v, "shape") else type(_v).__name__)
        for _k, _v in _dc.items()
    })

print("=== RECONSTRUCTION FUNCTION SOURCES ===")
for _name in ("rc_lin", "rc_mlp", "rc_norm", "rh_features", "head_features",
              "reconstruct_head", "rh_forward", "rh_prior"):
    _fn = globals().get(_name)
    if callable(_fn) and hasattr(_fn, "__code__"):
        try:
            print(_name, inspect.getsource(_fn)[:1800])
        except (OSError, TypeError):
            print(_name, "locals:", _fn.__code__.co_varnames,
                  "constants:", _fn.__code__.co_consts[:12])
```