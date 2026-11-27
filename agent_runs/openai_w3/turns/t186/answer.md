The records support an insulin effect of about −1.94 per unit and a norepinephrine effect of about +3.2 per rate unit, with different persistence. The original rollouts are much smaller. I’ll now recover the recurrent computation and readout inputs to calibrate an injection against those record-derived effects. Stable, drug-specific sensitivities would support a pilot edit.

```python
import inspect, numpy as np, torch
print("=== AVAILABLE FUNCTIONS ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    if inspect.isfunction(obj) and not key.startswith("_"):
        print(key, str(inspect.signature(obj)))

print("\n=== CORE TENSORS ===")
for name in NAMES:
    if name.startswith(("core.", "z_in.", "z_prior.", "z_head.")):
        print(name, SHAPES[name])

print("\n=== STATE / READOUT INPUT INVENTORY ===")
for key in ["S", "e80", "u80", "hflow81", "ctx81", "tokens145", "tokens154",
            "tokens155", "tokens_carrier155", "raw_carrier155", "raw_latent154",
            "view97", "views97", "ctx97", "z97", "dt97", "h97",
            "hbase105", "out105", "prior105", "contexts105", "readidx97",
            "reference", "ref6", "r67_probe_bias"]:
    if key not in globals():
        continue
    obj = globals()[key]
    if hasattr(obj, "files"):
        print(key, {k: obj[k].shape for k in obj.files})
    elif isinstance(obj, dict):
        print(key, {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                    for k, v in obj.items()})
    elif hasattr(obj, "shape"):
        print(key, tuple(obj.shape))
    else:
        print(key, type(obj).__name__, repr(obj)[:120])

for key in ["prelude155", "readout134", "fn160", "memory82"]:
    f = globals()[key]
    print("\n", key, "constants:", f.__code__.co_consts)
    try:
        print(inspect.getsource(f))
    except Exception:
        pass

print("\n=== STORED READOUT ARRAYS ===")
print([(key, tuple(obj.shape)) for key, obj in list(globals().items())
       if hasattr(obj, "shape") and (str(key).endswith(("97", "105", "121", "134")))
       and hasattr(obj, "ndim") and obj.ndim >= 1])
```
