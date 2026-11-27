The architecture confirms that time decay occurs before each bundle, and the records support drug-specific effects with uncertain persistence estimates. The recovery output was truncated before the channel details, so I have not changed anything. I’ll inspect the private coordinates and their recurrence coefficients; a persistence edit is justified only if those coordinates can retain dose history without altering ordinary state updates.

```python
import re
from pathlib import Path
import numpy as np
import torch

_notes259 = Path("/work/notes.md").read_text()
_sections259 = re.split(r"(?m)(?=^## )", _notes259)
_matches259 = [s for s in _sections259
               if re.search(r"reserved|private|channel|isolation|transport", s, re.I)]
print("=== CHANNEL CONSTRUCTION NOTES ===")
for _s259 in _matches259[:8]:
    print(_s259[:650])

def _np259(name, original=False):
    x = load_original(name) if original else load(name)
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

_f259 = _np259("core.flow.a").reshape(-1)
_fo259 = _np259("core.flow.a", True).reshape(-1)
_priv259 = np.flatnonzero(np.abs(_f259 - _fo259) > 1e-6)
print("\n=== MODIFIED FLOW COORDINATES ===", _priv259)
print("current", _f259[_priv259], "original", _fo259[_priv259])
print("flow shape", SHAPES["core.flow.a"])

if len(_priv259) <= 16:
    for _n259 in ("core.delta_in", "core.a_in", "core.out_norm.weight",
                  "core.out_norm.bias", "core.prelude_norm.weight",
                  "core.prelude_norm.bias"):
        _x259 = _np259(_n259)
        print(_n259, _x259.shape, _x259.reshape(-1)[_priv259])
    for _n259 in ("core.b_in.weight", "core.c_out.weight"):
        _x259 = _np259(_n259)
        _ordinary259 = np.setdiff1d(np.arange(_x259.shape[0]), _priv259)
        print(_n259, "private block", _x259[np.ix_(_priv259, _priv259)])
        print("cross maxima",
              np.max(np.abs(_x259[np.ix_(_priv259, _ordinary259)])),
              np.max(np.abs(_x259[np.ix_(_ordinary259, _priv259)])))
    print("slots private", _np259("core.slots")[:, _priv259])

print("\n=== KERNEL OBJECT NAMES ===")
_objects259 = []
for _n259, _v259 in list(globals().items()):
    if _n259.startswith("_") or _n259 in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(_v259, (np.ndarray, torch.Tensor)):
        _objects259.append(f"{_n259}:{tuple(_v259.shape)}")
    elif isinstance(_v259, dict):
        _objects259.append(f"{_n259}:dict({len(_v259)})")
    elif type(_v259).__name__ in {"DataFrame", "Series"}:
        _objects259.append(f"{_n259}:{tuple(_v259.shape)}")
    elif callable(_v259) and getattr(_v259, "__module__", "") == "__main__":
        _objects259.append(f"{_n259}()")
print(" | ".join(_objects259)[:2800])
```