The last check rules out stale weights, but the CPU reconstruction still misses the ordinary state despite closely matching parts of the private dose channel. That makes its asymmetry diagnosis unreliable. I’ll recover the architecture and diagnostic helpers next; agreement with saved GPU states would validate the reconstruction, while continued disagreement would refute it.

```python
import inspect, pathlib, types
print("=== ARCHITECTURE ===")
print(pathlib.Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(pathlib.Path("/data/README.md").read_text())

notes_text = pathlib.Path("/work/notes.md").read_text()
print("\n=== NOTEBOOK: EARLIER IDENTIFICATION AND EDITS ===")
print(notes_text if len(notes_text) <= 26000 else notes_text[:18000] + "\n[...]\n" + notes_text[-8000:])

print("\n=== PERSISTENT FUNCTIONS ===")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and not k.startswith("_"):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, sig)

print("\n=== PERSISTENT DATA / SETTINGS ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or isinstance(v, types.ModuleType) or callable(v):
        continue
    if hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        ks = list(v.keys())
        print(k, "dict", len(v), "keys:", str(ks[:14])[:700])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v), str(v[:8])[:350])
    elif k.startswith(("wm_", "causal", "channel", "head", "strength", "rate", "sensor")):
        print(k, repr(v)[:500])
print("\n=== CURRENT EDIT COUNT ===", len(edited()))
```
