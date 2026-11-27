The last test reduced the local prediction error, but it still does not reproduce the forward tool. I’ll check the architecture and reconstruction code before using local scores to protect factual fit. Matching the original predictions would validate those scores; a remaining mismatch would mean more debugging is needed.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RECONSTRUCTION FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and any(s in name.lower() for s in
        ("head", "core", "register", "factual", "feature", "prior", "act0")):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig, "constants:", str(obj.__code__.co_consts)[:650])

print("\n=== RECENT DIAGNOSTIC OBJECTS ===")
for name in sorted(list(globals())):
    if any(name.endswith(str(k)) for k in range(94, 106)):
        obj = globals()[name]
        if hasattr(obj, "shape"):
            desc = f"{type(obj).__name__} shape={tuple(obj.shape)}"
        elif isinstance(obj, dict):
            desc = f"dict keys={list(obj)[:18]}"
        elif isinstance(obj, (list, tuple)):
            desc = f"{type(obj).__name__} length={len(obj)}"
        else:
            continue
        print(name, desc)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 106: exact reconstruction recovery\n"
            "- Last-numeric layout changes reduced but did not eliminate the external mismatch.\n"
            "- Re-reading the architecture and recovering local reconstruction interfaces before trusting any edit score.\n"
            "- Python runs: 106; forward passes: 4; no submitted edits.\n")
```