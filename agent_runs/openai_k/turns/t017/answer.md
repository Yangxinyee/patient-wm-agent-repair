The reconstructed posterior still differs substantially from the saved states, so it is not yet a safe basis for editing weights. I’ll inspect the exact architecture and retained helpers to locate the mismatch. Agreement with saved states would validate the reconstruction; persistent disagreement would rule it out.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== TENSOR SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== RETAINED VARIABLES AND HELPERS ===")
for key, value in list(globals().items()):
    if key.startswith("_"):
        continue
    if inspect.isfunction(value):
        try:
            signature = str(inspect.signature(value))
        except Exception:
            signature = "?"
        print("FUNCTION", key, signature)
    elif hasattr(value, "shape"):
        print("ARRAY", key, type(value).__name__, value.shape)
    elif isinstance(value, dict):
        print("DICT", key, "keys:", list(value)[:25])
    elif isinstance(value, (list, tuple)) and len(value) > 0:
        print("SEQUENCE", key, "length:", len(value))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 12: reconstruction audit\n"
            "- The posterior reconstruction remains inaccurate; no edit is justified by it yet.\n"
            "- Re-reading the exact architecture and inventorying retained helpers before further reconstruction.\n"
            "- Python runs: 12; forward passes: 4; no weights edited.\n")
```