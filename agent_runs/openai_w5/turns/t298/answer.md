The notes confirm that the current repair uses short-lived dose signals and sensor-specific value corrections; it was not designed to accumulate exposure. The last printout was truncated by a function’s default arguments, so infusion-end handling remains unresolved. I’ll inspect the actual records and helper code without printing those defaults. If end records distort the dose signal, that would refute persistence as the first repair to make.

```python
import inspect
import numpy as np
import torch

print("=== RECORD DOCUMENTATION ===")
print(Path("/data/README.md").read_text()[:2200])
print("\n=== SAMPLE RECORD ===")
print(train.head(16).to_string(index=False))

print("\n=== ACTION HELPER SOURCE, WITHOUT DEFAULT-ARGUMENT REPR ===")
for _name261 in ("drug_atom205", "set205", "bounded_definition"):
    if _name261 in globals():
        _fn261 = globals()[_name261]
        print(_name261)
        try:
            print(inspect.getsource(_fn261)[:1600])
        except Exception:
            print("arguments", _fn261.__code__.co_varnames[:_fn261.__code__.co_argcount])
            print("referenced names", _fn261.__code__.co_names)
            print("constants", str(_fn261.__code__.co_consts)[:450])

print("\n=== RECENT DIAGNOSTIC OBJECT NAMES ===")
_recent261 = []
for _name261, _val261 in list(globals().items()):
    if _name261.startswith(("r251", "r252", "r253", "r254", "r255", "r256", "r257")):
        if isinstance(_val261, dict):
            _recent261.append(f"{_name261}: keys={list(_val261)[:10]}")
        elif hasattr(_val261, "shape"):
            _recent261.append(f"{_name261}: shape={tuple(_val261.shape)}")
        elif callable(_val261):
            _recent261.append(f"{_name261}: callable")
        elif isinstance(_val261, (int, float, str, list, tuple)):
            _recent261.append(f"{_name261}: {str(_val261)[:100]}")
print("\n".join(_recent261)[:1300])

print("\n=== SENSOR HEAD PRIVATE COEFFICIENTS ===")
_hw261 = _np259("attr_heads.value_bin.0.weight")
_hb261 = _np259("attr_heads.value_bin.0.bias")
_ow261 = _np259("attr_heads.value_bin.2.weight")
for _sensor261, (_plus261, _minus261) in sensor_neurons.items():
    print(_sensor261, (_plus261, _minus261),
          "bias", _hb261[[_plus261, _minus261]],
          "private input", _hw261[np.ix_([_plus261, _minus261], _priv259)])
    print("output column norms", np.linalg.norm(_ow261[:, [_plus261, _minus261]], axis=0))
```