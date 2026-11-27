The records support insulin lowering next-hour glucose by about 2 per rate unit and norepinephrine raising MAP by about 3, contrary to the naive associations. These are proximal estimates, not yet rollout targets. The output was cut off before the implementation inventory; I’ll retrieve that compactly so we can build an edit without repeating the encoder investigation.

```python
import numpy as np, pandas as pd, torch

_items94 = list(globals().items())
print("TABLES")
for _k, _v in _items94:
    if isinstance(_v, pd.DataFrame):
        print(_k, _v.shape, "|", ",".join(map(str, _v.columns[:14])))

print("\nFUNCTIONS")
print(" | ".join(_k for _k, _v in _items94
                 if callable(_v) and getattr(_v, "__module__", None) == "__main__"))

print("\nARRAYS")
_lines94 = []
for _k, _v in _items94:
    if isinstance(_v, (np.ndarray, torch.Tensor)) and _v.ndim >= 2:
        if _v.shape[0] >= 100:
            _lines94.append(f"{_k}: {tuple(_v.shape)} {str(_v.dtype)}")
print("\n".join(_lines94)[:2400])

print("\nSMALL CONFIG/STATE OBJECTS")
for _k, _v in _items94:
    if isinstance(_v, dict) and not _k.startswith("__"):
        _keys = list(_v)
        if any(s in _k.lower() for s in ("state", "weight", "cache", "reg", "vocab", "fit", "caus", "dose", "result")):
            print(_k, len(_keys), str(_keys[:12])[:230])

print("\nRELEVANT TENSOR NAMES")
for _n in NAMES:
    if any(s in _n for s in ("z_post", "z_prior", "z_in", "z_head",
                             "attr_heads.value_bin", "query_norm", "last_numeric")):
        print(_n, SHAPES[_n])

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 94 — compact implementation inventory\n"
             "- Architecture recovered. Record-based proximal estimates remain approximately "
             "insulin -2 and norepinephrine +3 per rate unit.\n"
             "- Retrieving retained numerical helpers and states with bounded output; "
             "no weights changed.\n")
```