The remaining decoder error is almost perfectly explained by register features. That points to an input-construction bug, not a reason to change more weights. I’ll inspect the architecture and current helpers; calibration is safe only once the local decoder reproduces the GPU results.

```python
from pathlib import Path
import re, inspect

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRECENT OBJECTS")
for k, v in sorted(list(globals().items())):
    if re.search(r"(16[4-9])$", k) or (
        callable(v) and re.search(r"query|static|decoder|register|input|prob", k, re.I)
    ):
        print(k, type(v).__name__, getattr(v, "shape", ""),
              list(v.keys())[:12] if isinstance(v, dict) else "")
        if callable(v) and hasattr(v, "__code__"):
            print("  args:", inspect.signature(v), "names:", v.__code__.co_names)

print("\nEARLY EVIDENCE / BASELINE NOTES")
nt170 = Path("/work/notes.md").read_text()
for line in nt170.splitlines():
    if re.search(r"true effect|causal|instrument|random|baseline77|target.*insulin|insulin.*target", line, re.I):
        print(line[:1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 170: targeted decoder recovery\n"
            "Run169's static residual is explained to numerical precision by register features. "
            "Inspecting architecture and retained helpers before correcting assembly; no weights changed. "
            "Python runs: 170; forwards: 22.\n")
```