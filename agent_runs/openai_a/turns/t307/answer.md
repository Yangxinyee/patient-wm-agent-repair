The last run failed because I used the wrong key for the bin definitions; it gave no calibration evidence and changed no weights. I’ll recover the correct bin structure and existing helpers before testing another edit. If they support a reliable calibration calculation, I can test it; otherwise I’ll retain the validated checkpoint.

```python
from pathlib import Path
import inspect, json

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD COLUMNS ===")
print(Path("/data/README.md").read_text())

_recovery_notes = Path("/work/notes.md").read_text()
print("=== NOTEBOOK START; total characters =", len(_recovery_notes), "===")
print(_recovery_notes[:14000])

print("=== VOCAB STRUCTURE ===")
for _k, _v in VOCAB.items():
    print(_k, type(_v).__name__,
          list(_v.keys())[:12] if isinstance(_v, dict) else
          f"length={len(_v)}" if hasattr(_v, "__len__") else repr(_v))
print("=== GLUCOSE BIN ENTRIES ===")
for _k, _v in VOCAB.items():
    if isinstance(_v, list):
        for _entry in _v:
            if isinstance(_entry, dict) and "glucose_finger" in json.dumps(_entry):
                print(_k, _entry)
    elif isinstance(_v, dict):
        for _ek, _ev in _v.items():
            if "glucose_finger" in str(_ek) or (
                isinstance(_ev, dict) and "glucose_finger" in json.dumps(_ev)
            ):
                print(_k, _ek, _ev)

print("=== USER-DEFINED HELPERS ===")
for _name, _obj in sorted(list(globals().items())):
    if inspect.isfunction(_obj) and getattr(_obj, "__module__", "") == "__main__":
        try:
            print(_name, inspect.signature(_obj))
        except Exception:
            print(_name)

print("=== RELEVANT PERSISTENT OBJECTS ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_"):
        continue
    if any(_part in _name.lower() for _part in
           ("checkpoint", "cache", "calib", "ridge", "gate", "rd_", "panel", "target", "tilt")):
        if isinstance(_obj, dict):
            print(_name, "dict", list(_obj.keys())[:18])
        elif hasattr(_obj, "shape"):
            print(_name, type(_obj).__name__, tuple(_obj.shape))
        elif isinstance(_obj, (str, int, float, list, tuple)):
            print(_name, repr(_obj)[:240])
print("=== CURRENT EDITS ===")
print(edited())
```
