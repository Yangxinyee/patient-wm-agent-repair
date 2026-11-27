The recovered notes show a fairly stable insulin estimate but sharply changing norepinephrine estimates under different adjustments. That makes the raw norepinephrine association unreliable. I’ll recover the stronger identification results and the current edit builder; agreement across identification checks would support a target, while disagreement would limit what we can claim.

```python
import re, inspect, dis
from pathlib import Path

_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"(?m)^## ", _notes)
print("=== IDENTIFICATION NOTES ===")
_chosen = []
for s in _sections:
    title = s.splitlines()[0] if s.splitlines() else ""
    if any(t in title.lower() for t in
           ["discontinu", "identif", "threshold", "causal", "persistence", "norepinephrine"]):
        _chosen.append(s)
print("\n".join(_chosen)[-4800:])

print("\n=== CANDIDATE FUNCTION NAMES ===")
_fnames = []
for k, v in list(globals().items()):
    if callable(v) and any(t in k.lower() for t in
                          ["install", "build", "gated", "r3", "activity"]):
        try:
            print(k, inspect.signature(v))
            _fnames.append(k)
        except (ValueError, TypeError):
            pass

print("\n=== RELEVANT DATA OBJECTS ===")
for k, v in list(globals().items()):
    if not callable(v) and not k.startswith("__") and any(t in k.lower() for t in
        ["checkpoint", "gated", "r3", "activity", "causal", "rd_", "target"]):
        if isinstance(v, dict):
            print(k, "dict", list(v)[:12])
        elif hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, (int, float, str)):
            print(k, repr(v)[:100])

print("\n=== BUILDER BYTECODE ===")
for k in _fnames:
    if any(t in k.lower() for t in ["install", "gated", "candidate"]):
        f = globals()[k]
        if hasattr(f, "__code__"):
            print("\n", k)
            try:
                print(inspect.getsource(f))
            except (OSError, TypeError):
                print("names:", f.__code__.co_names,
                      "locals:", f.__code__.co_varnames,
                      "constants:", repr(f.__code__.co_consts)[:600])
```
