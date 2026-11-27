There is an important discrepancy: `last_numeric.weight` takes eight inputs, although the architecture describes only four. That makes the register encoding a concrete suspect. I’ll inspect the reconstruction function and metadata before editing; finding the correct eight-field layout should improve both predicted means and NLL.

```python
import inspect, dis, io, contextlib
import numpy as np

print("=== RECONSTRUCTION FUNCTIONS ===")
for _fname in ["reading_metadata", "make_head_features", "lin", "ln", "mlp"]:
    _fn = globals()[_fname]
    print("\n", _fname)
    try:
        print(inspect.getsource(_fn)[:2600])
    except Exception:
        print("constants:", str(_fn.__code__.co_consts)[:1200])
        if _fname == "make_head_features":
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                dis.dis(_fn)
            print(_buf.getvalue()[:4000])

print("\n=== HEAD METADATA ===")
print(head_meta_tune.columns.tolist())
print(head_meta_tune.head(2).to_string(index=False))
print("\nWeights cached:", list(W))
print("\n=== STATES ===")
for _key, _obj in state_files.items():
    print(_key, type(_obj).__name__)
    if isinstance(_obj, (str, bytes)):
        _obj = np.load(_obj)
    _keys = list(_obj.files) if hasattr(_obj, "files") else list(_obj)
    print("keys:", _keys)
    for _k in _keys:
        _v = _obj[_k]
        print(_k, getattr(_v, "shape", None), getattr(_v, "dtype", None))

print("\n=== DATA / VOCAB DETAILS ===")
print("events:", events)
print("columns:", columns)
print("train IDs:", train_ids[:5], "tuning IDs:", tuning_ids[:5])
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("numeric note:", VOCAB["numeric_note"])
print("numeric structure:", type(VOCAB["numeric"]).__name__)
print("normalisation entries:", str(norms)[:1100])
print("Current edits:", edited())
```