The architecture confirms that dose magnitude enters through numeric and bin embeddings, separately from drug identity. Earlier analyses found little stable dose-response signal, but I still need the edit’s validation results before deciding to keep it. I’ll recover those results and the retained analysis objects; a factual NLL increase above 0.005 would rule out the current edit.

```python
from pathlib import Path
import numpy as np
import inspect

_notes = Path("/work/notes.md").read_text()
_start = _notes.find("## Run 8")
_stop = _notes.find("## Run 13")
if _stop < 0:
    _stop = max(_start, len(_notes) - 8000)
print("=== NOTEBOOK MIDDLE ===")
print(_notes[_start:_stop][:5500])

print("=== CURRENT EDITS ===")
for _name in edited():
    _a = np.asarray(load(_name))
    _b = np.asarray(load_original(_name))
    _d = _a - _b
    print(_name, _a.shape, "change_norm", round(float(np.linalg.norm(_d)), 6),
          "changed_entries", int(np.count_nonzero(_d)))
    if _a.ndim == 2 and _a.shape[1] <= 8:
        print("original column norms:", np.linalg.norm(_b, axis=0).round(4))
        print("edited column norms:", np.linalg.norm(_a, axis=0).round(4))
    elif _a.ndim == 2 and _a.shape[0] <= 20:
        print("edited row norms:", np.linalg.norm(_a, axis=1).round(4))

print("=== RETAINED OBJECTS ===")
_items = []
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or _name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if inspect.isfunction(_obj):
        _items.append(f"{_name}{inspect.signature(_obj)}")
    elif hasattr(_obj, "shape"):
        _items.append(f"{_name}:{type(_obj).__name__}{_obj.shape}")
    elif isinstance(_obj, (list, dict, tuple, set)):
        _items.append(f"{_name}:{type(_obj).__name__}[{len(_obj)}]")
print("\n".join(_items)[:3000])

print("=== FORWARD FILES ===")
for _p in sorted(Path("/work").glob("fwd_*/*")):
    print(_p.name, _p.parent.name, _p.stat().st_size)
```
