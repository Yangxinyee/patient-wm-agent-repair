The inspection was truncated because a helper’s default argument printed the entire weight cache. It provides no new evidence about drug effects. I’ll print bounded summaries instead, so the reconstruction inputs are visible and we can test the suspected register mismatch directly.

```python
import inspect, json
import numpy as np
from pathlib import Path

_gitems = list(globals().items())

print("=== HELPERS (PARAMETER NAMES ONLY) ===")
for _nm, _ob in _gitems:
    if inspect.isfunction(_ob) and _nm not in {
        "load", "load_original", "save_edit", "drop_edit", "edited", "compact"
    }:
        print(_nm, list(inspect.signature(_ob).parameters))

print("\n=== RECENT OBJECT SUMMARIES ===")
for _nm, _ob in _gitems[-105:]:
    if _nm.startswith("_"):
        continue
    _desc = type(_ob).__name__
    if hasattr(_ob, "shape"):
        _desc += " shape=" + str(tuple(_ob.shape))
    elif isinstance(_ob, dict):
        _desc += " keys=" + str(list(_ob)[:6])
    elif isinstance(_ob, (list, tuple)):
        _desc += " length=" + str(len(_ob))
    elif hasattr(_ob, "files"):
        _desc += " keys=" + str(_ob.files)
    elif isinstance(_ob, (str, int, float, bool)):
        _desc += " " + repr(_ob)[:120]
    print(_nm, _desc)

print("\n=== FORWARD FILES ===")
for _folder in sorted(Path("/work").glob("fwd_*")):
    print(_folder.name, [p.name for p in _folder.iterdir()])

print("\n=== VOCAB ===")
print("keys:", list(VOCAB))
for _key in ("config", "codes"):
    print(_key, json.dumps(VOCAB.get(_key, {}))[:1400])
for _key in VOCAB:
    if any(s in _key.lower() for s in ("norm", "scal", "categor", "unit")):
        print(_key, str(VOCAB[_key])[:800])

print("\n=== GAP / REGISTER TENSORS ===")
for _nm in NAMES:
    if _nm.startswith(("gap.", "last_", "monitor")):
        print(_nm, SHAPES[_nm])
```
