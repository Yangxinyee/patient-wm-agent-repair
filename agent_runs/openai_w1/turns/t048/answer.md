The states encode recorded drug rates well, but the earlier value-head features did not recover dose innovations on held-out stays. That argues against simply amplifying raw dose signals. I’ll inspect the posterior-update features next; reliable innovation decoding with little activation on non-drug bundles would support a more selective edit.

```python
import inspect
import dis
import io
import contextlib

print("=== RUN-36 OBJECTS ===")
for _name, _obj in sorted(list(globals().items())):
    if not _name.startswith("r36"):
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict", len(_obj), repr(list(_obj.items())[:1])[:350])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj), repr(_obj[:2])[:350])
    elif not callable(_obj):
        print(_name, repr(_obj)[:180])

print("\n=== INNOVATION OBJECT STRUCTURE ===")
for _key, _obj in causal_residuals.items():
    print(_key, type(_obj).__name__)
    if isinstance(_obj, dict):
        for _k, _v in _obj.items():
            print(" ", _k, type(_v).__name__, getattr(_v, "shape", None))
    elif isinstance(_obj, (list, tuple)):
        print([(type(_v).__name__, getattr(_v, "shape", None)) for _v in _obj])

print("\n=== ATOM / SET ENCODER IMPLEMENTATIONS ===")
for _fn in (r36_atom, r36_set_encode):
    print("\nFUNCTION", _fn.__name__)
    try:
        print(inspect.getsource(_fn)[:2800])
    except (OSError, TypeError):
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            dis.dis(_fn)
        print(_buf.getvalue()[:2800])

print("\n=== FULL-STATE RATE DECODING RESULTS ===")
print(repr(decode_results[-6:])[:1800])
print("\n=== CAUSAL ESTIMATE SUMMARY ===")
for _v in causal_estimates:
    if isinstance(_v, dict):
        print({k: _v[k] for k in ("event", "drug", "subset", "beta", "cluster_se")
               if k in _v})
```