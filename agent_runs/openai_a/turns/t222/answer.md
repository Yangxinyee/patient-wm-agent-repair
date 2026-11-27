The trial is installed, but its cached factual loss is higher; it is not yet safe to accept. Earlier cached losses differed substantially from GPU results, so I will recover the architecture, checkpoints, and stay IDs before validating it. Acceptance requires stronger insulin responses without exceeding the 0.005-nat factual-loss limit.

```python
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

_notes_text = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK: EVIDENCE AND EDIT DESIGN ===")
print(_notes_text[:18000])

print("=== RELEVANT PERSISTENT VARIABLES ===")
_keys = ("checkpoint", "rollback", "r3", "gate", "separator",
         "ids", "cache", "contrast", "head", "rd", "causal")
for _name in sorted(list(globals())):
    if _name.startswith("_") or not any(k in _name.lower() for k in _keys):
        continue
    _obj = globals()[_name]
    _shape = getattr(_obj, "shape", None)
    if callable(_obj):
        import inspect
        try:
            _detail = str(inspect.signature(_obj))
        except Exception:
            _detail = ""
    elif isinstance(_obj, dict):
        _detail = "keys=" + repr(list(_obj.keys())[:20])
    elif isinstance(_obj, (list, tuple)):
        _detail = "len=" + str(len(_obj))
        if "ids" in _name.lower():
            _detail += " first=" + repr(_obj[:8])
    elif isinstance(_obj, (str, int, float, bool)):
        _detail = repr(_obj)[:200]
    else:
        _detail = ""
    print(_name, type(_obj).__name__, _shape, _detail)

print("=== CURRENT EDITS ===")
print(edited())

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Restart audit; Python 197, forwards 16/50\n"
        "- Current finger-selective scale-3 trial remains unaccepted.\n"
        "- Recovering architecture, causal evidence, checkpoints, and IDs "
        "before GPU factual and rollout validation.\n"
    )
```