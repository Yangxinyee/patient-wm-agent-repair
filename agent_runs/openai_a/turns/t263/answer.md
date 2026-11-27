The edit uses a gated logit tilt, so a separate record-fitted calibration may reduce its factual cost. I first need to locate the factual caches and distinguish them from synthetic dose contrasts. A useful calibration must improve held-out reading likelihood—not merely fit the synthetic examples.

```python
import dis, inspect, numpy as np, pandas as pd, torch

print("GRAFT CONSTANTS")
for _n in ["sc_names", "sc_nodes", "sc_rank", "selective_sep_w", "selective_sep_b"]:
    _v = globals().get(_n)
    print(_n, tuple(_v.shape) if hasattr(_v, "shape") else _v)

print("\nMAKE_CONTRAST_GRAFT")
for _i in list(dis.get_instructions(make_contrast_graft))[:135]:
    print(_i.opname, _i.argrepr)

print("\nCACHE INVENTORY")
_recovery_caches = {}
_seen_cache = set()
def _find_recovery_cache(_obj, _path, _depth=0):
    if id(_obj) in _seen_cache:
        return
    if isinstance(_obj, dict):
        _seen_cache.add(id(_obj))
        if "x" in _obj and "logits" in _obj and hasattr(_obj["x"], "shape"):
            _recovery_caches[_path] = _obj
            _meta = _obj.get("meta")
            _events = _meta["event"].value_counts().to_dict() if isinstance(_meta, pd.DataFrame) and "event" in _meta else None
            print(_path, "x", _obj["x"].shape, "nll", _obj.get("original_nll"), "events", _events)
        elif _depth < 2:
            for _k, _v in list(_obj.items()):
                if isinstance(_v, (dict, list, tuple)):
                    _find_recovery_cache(_v, f"{_path}[{_k!r}]", _depth + 1)
    elif isinstance(_obj, (list, tuple)) and _depth < 2:
        for _j, _v in enumerate(_obj):
            if isinstance(_v, dict):
                _find_recovery_cache(_v, f"{_path}[{_j}]", _depth + 1)

for _n, _v in list(globals().items()):
    if not _n.startswith("_") and _n != "ns_snapshot" and isinstance(_v, (dict, list, tuple)):
        _find_recovery_cache(_v, _n)

print("\nOTHER SYNTHETIC/CACHE OBJECTS")
for _n, _v in list(globals().items()):
    if _n.startswith(("sc_", "r3_", "selective_", "r140_", "repair223_", "repair224_")):
        if hasattr(_v, "shape"):
            print(_n, type(_v).__name__, tuple(_v.shape))
        elif isinstance(_v, (dict, list, tuple)):
            print(_n, type(_v).__name__, len(_v))
        elif isinstance(_v, (int, float, str)):
            print(_n, repr(_v)[:100])
```