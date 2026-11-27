The glucose bins have unequal widths, so a tilt by bin rank is not equivalent to a shift in glucose units. I’ll inspect the graft formula and changed neurons next. That will show whether I can add a glucose-specific calibration adjustment; any improvement must then survive the exact GPU factual and rollout tests.

```python
import inspect, dis, io
print("=== SELECTED HELPERS ===")
for _name in ("make_selective_contrast_graft", "cache_losses",
              "cpu_head_probs", "head_average_probs3", "screen_value_candidate"):
    _obj = globals()[_name]
    print(_name, inspect.signature(_obj))
    try:
        _src = inspect.getsource(_obj)
        print(_src[:4200] if _name == "make_selective_contrast_graft" else _src[:1300])
    except Exception:
        if _name == "make_selective_contrast_graft":
            _buf = io.StringIO()
            dis.dis(_obj, file=_buf)
            print(_buf.getvalue()[:4400])
        else:
            print("names:", _obj.__code__.co_names,
                  "constants:", repr(_obj.__code__.co_consts)[:650])

print("=== CONFIGURATIONS ===")
for _name in ("activity_gate_config", "gate035_config", "cfg",
              "protected_separators", "activity_graft_rows",
              "selective_trial_scale", "selective_trial_lam"):
    if _name in globals():
        print(_name, repr(globals()[_name])[:750])

print("=== EDITED ROWS ===")
for _name in edited():
    _a = load(_name).numpy() if hasattr(load(_name), "numpy") else np.asarray(load(_name))
    _o = load_original(_name).numpy() if hasattr(load_original(_name), "numpy") else np.asarray(load_original(_name))
    _d = _a - _o
    print(_name, _a.shape, "max difference", np.max(np.abs(_d)))
    if _d.ndim == 2:
        _axis = 1 if ".0." in _name else 0
        _indices = np.flatnonzero(np.max(np.abs(_d), axis=_axis) > 1e-7)
        print("changed rows/columns:", _indices.tolist()[:30])
    else:
        print("changed entries:", np.flatnonzero(np.abs(_d)>1e-7).tolist()[:30])
print("Matches validated gated8:",
      all(np.array_equal(np.asarray(load(_n)), np.asarray(_v))
          for _n, _v in validated_gated8_checkpoint.items()))

print("=== CACHE STRUCTURE ===")
for _name in ("calibration_cache", "sc_fact", "sc_syn"):
    _obj = globals().get(_name)
    print(_name, type(_obj).__name__)
    if isinstance(_obj, dict):
        for _k, _v in list(_obj.items())[:12]:
            if isinstance(_v, dict):
                print(_k, "dict", list(_v.keys())[:10])
            elif hasattr(_v, "shape"):
                print(_k, type(_v).__name__, tuple(_v.shape))
                if hasattr(_v, "columns"):
                    print("columns", list(_v.columns))
            else:
                print(_k, repr(_v)[:100])
print("rank_recovery:", repr(globals().get("rank_recovery"))[:300])
```
