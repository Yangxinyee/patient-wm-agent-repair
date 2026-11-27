The architecture confirms that dose edits change every drug atom, including END atoms. The inspection output was truncated before the transport details appeared, so it does not yet establish where the asymmetry arises. I’ll recover a compact inventory and the relevant notebook entries without printing large tensor defaults.

```python
import inspect, re
from pathlib import Path
import numpy as np, torch

_g252 = dict(globals())
print("=== FUNCTION INVENTORY (NO DEFAULT VALUES) ===")
for _k, _v in sorted(_g252.items()):
    if inspect.isfunction(_v) and getattr(_v, "__module__", "") == "__main__" and not _k.startswith("_fn"):
        _c = _v.__code__
        print(_k, "args", _c.co_varnames[:_c.co_argcount],
              "refs", _c.co_names[:18])

print("\n=== OBJECT INVENTORY ===")
_lines252 = []
for _k, _v in sorted(_g252.items()):
    if re.search(r"(action|transport|center|candidate|causal|panel|coef|gate|pair|exposure|cache|state|tune|reserved|sensor)", _k, re.I):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            _lines252.append(f"{_k}: {type(_v).__name__} {tuple(_v.shape)}")
        elif isinstance(_v, dict):
            _lines252.append(f"{_k}: dict keys={list(_v)[:16]}")
        elif isinstance(_v, (list, tuple)):
            _lines252.append(f"{_k}: {type(_v).__name__} len={len(_v)}")
        elif isinstance(_v, (int, float, str, bool)):
            _lines252.append(f"{_k}: {str(_v)[:120]}")
        elif hasattr(_v, "shape"):
            _lines252.append(f"{_k}: {type(_v).__name__} shape={_v.shape}")
print("\n".join(_lines252)[:3300])

_notes252 = Path("/work/notes.md").read_text()
_sections252 = re.split(r"(?m)(?=^## )", _notes252)
print("\n=== IDENTIFICATION / TRANSPORT NOTES ===")
_sel252 = [s for s in _sections252
           if re.search(r"(identification|causal estim|negative control|instrument|transport design|private channel|end atom|world atom)", s, re.I)]
print("\n".join(s[:1300] for s in _sel252[:5])[:4200])
print("\n=== CURRENT EDIT COUNT ===", len(edited()))
```