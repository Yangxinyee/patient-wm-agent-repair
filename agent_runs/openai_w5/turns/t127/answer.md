The records support provisional effects of roughly −1.8 glucose units per insulin unit and +3 MAP units per norepinephrine rate unit at the next hour. The architecture confirms that a repair needs a recurrent dose pathway, not just a value-head adjustment. I’ll recover the retained helpers and recent fits; accurate decoding on unseen doses is necessary before installing that pathway.

```python
import inspect
from pathlib import Path

_g113 = list(globals().items())
_funcs113 = [
    (n, v) for n, v in _g113
    if not n.startswith("_") and inspect.isfunction(v)
    and getattr(v, "__module__", "") == "__main__"
]
print("=== FUNCTION NAMES ===")
print(", ".join(n for n, v in _funcs113))

print("\n=== RECENT OBJECTS ===")
_recent113 = [
    (n, v) for n, v in _g113
    if not n.startswith("_")
    and any(n.endswith(str(k)) for k in range(100, 112))
]
for n, v in _recent113[:90]:
    if inspect.isfunction(v):
        print(n, "function", str(inspect.signature(v)))
    elif isinstance(v, dict):
        print(n, "dict", list(v)[:15])
    elif isinstance(v, (list, tuple)):
        print(n, type(v).__name__, "len", len(v), repr(v[:3])[:140])
    elif hasattr(v, "shape") and not callable(v.shape):
        print(n, type(v).__name__, tuple(v.shape))
    else:
        print(n, type(v).__name__, repr(v)[:140])

print("\n=== PANEL COLUMNS ===")
for n in ["panel", "panel_tune"]:
    if n in globals():
        print(n, list(globals()[n].columns))

_notes113 = Path("/work/notes.md").read_text()
print("\n=== EARLIER CAUSAL VALIDATION NOTES ===")
_a113 = _notes113.find("## Sensor checks")
_b113 = _notes113.find("##", _a113 + 4000)
print(_notes113[_a113: min(_a113 + 4300, len(_notes113))])

print("\n=== RELEVANT TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in [
        "encode_action", "embed.numeric", "core.c_out", "core.b_in",
        "core.delta_in", "core.a_in", "core.out_norm",
        "attr_heads.value_bin", "last_numeric"
    ]):
        print(n, SHAPES[n])
```
