The architecture confirms that dose edits affect recurrent state, while generated readings feed back through the last-value register. That explains why a local head contrast is insufficient. The output was truncated, so I’ll retrieve only the key identification results and helper names; these should establish both the causal target and how to reproduce the protected edit.

```python
import re
from pathlib import Path

_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"(?m)^## ", _notes)
for _term in ["threshold identification", "Sustained insulin", "Residual-dose decoder",
              "Latent-robust gated graft", "Feedback diagnosis", "World-embedding"]:
    _matches = [s for s in _sections if _term.lower() in s.split("\n", 1)[0].lower()]
    for _s in _matches[:1]:
        print("\nSECTION:", _s[:1050])

print("\nUSER FUNCTIONS")
_fn = []
for _n, _v in list(globals().items()):
    if not _n.startswith("_") and callable(_v) and getattr(_v, "__module__", "") == "__main__":
        _fn.append(_n)
print(", ".join(sorted(_fn)))

print("\nRELEVANT OBJECTS")
_rows = []
for _n, _v in list(globals().items()):
    if _n.startswith("_") or callable(_v) or not re.search(
        r"r3|cache|checkpoint|contrast|ids|policy|beta|persist|ridge|gate|finger", _n, re.I
    ):
        continue
    if hasattr(_v, "shape"):
        _rows.append(f"{_n}: {type(_v).__name__} {tuple(_v.shape)}")
    elif isinstance(_v, dict):
        _rows.append(f"{_n}: dict({len(_v)}), keys={list(_v)[:6]}")
    elif isinstance(_v, (float, int, str)):
        _rows.append(f"{_n}: {repr(_v)[:100]}")
    elif isinstance(_v, (list, tuple)):
        _rows.append(f"{_n}: {type(_v).__name__}({len(_v)}), first={repr(_v[:3])[:100]}")
print("\n".join(_rows)[:4700])
print("\nCURRENT EDITS:", edited())
```