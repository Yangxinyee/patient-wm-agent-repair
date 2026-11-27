The core reconstruction still disagrees with the saved states, so it is not yet safe to derive an edit from it. No weights have changed. I’ll recover the architecture and earlier record-based findings; close state reconstruction and consistent causal estimates would support proceeding, while discrepancies would require revising the approach.

```python
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(notes_text[:-8000] if len(notes_text) > 8000 else notes_text)

print("\n=== KERNEL INVENTORY ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or callable(obj):
        continue
    typ = type(obj).__name__
    if typ in ("ndarray", "Tensor", "DataFrame", "Series"):
        print(name, typ, getattr(obj, "shape", None))
        if typ == "DataFrame":
            print("  columns:", list(obj.columns))
    elif typ == "dict":
        keys = list(obj.keys())
        print(name, typ, len(obj), "keys:", keys[:18])
    elif typ in ("list", "tuple") and len(obj) > 0:
        print(name, typ, len(obj), "first element type:", type(obj[0]).__name__)

print("\n=== CURRENT EDITS ===")
print(edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 28: restart recovery\n"
            "- Core reconstruction in run 27 remains unvalidated; no edit is justified from it yet.\n"
            "- Recovering architecture, prior causal evidence, and persistent kernel objects.\n"
            "- Python runs: 28; forward passes: 4; no weights edited.\n")
```