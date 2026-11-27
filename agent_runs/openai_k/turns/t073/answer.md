The reserved features reconstruct the original posterior very closely, but their dose-centering errors could still hurt factual predictions. I’ll recover the architecture and earlier causal estimates before applying an edit. The hypothesis will hold only if a dose-sensitive correction improves rollouts while preserving factual NLL.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
notes = Path("/work/notes.md").read_text()
print(notes[:22000])
print("\n=== RECENT OBJECTS ===")
for name in sorted(list(globals())):
    if name.endswith(("65", "66", "_candidates")) or name.startswith(("D65", "J65", "core", "probe", "posterior")):
        obj = globals()[name]
        if callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                print(name, type(obj).__name__)
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:30])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 67: restart recovery\n"
            "- Carrier features preserve posterior outputs locally, but factual safety and dynamic effects remain untested.\n"
            "- Recovering architecture, original identification evidence, and candidate interfaces before an actual edit.\n"
            "- Python runs: 67; forward passes: 4; no submitted edits.\n")
```