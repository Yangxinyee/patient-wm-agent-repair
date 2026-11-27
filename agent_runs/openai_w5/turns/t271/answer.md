```python
from pathlib import Path
print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== NOTEBOOK ===")
print(Path("/work/notes.md").read_text())
print("=== KERNEL OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), list(v.keys())[:12])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v), repr(v[:5])[:300])
    elif callable(v):
        print(k, "callable")
print("=== VOCAB KEYS ===", list(VOCAB))
print("=== DATA README ===")
print(Path("/data/README.md").read_text())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart after forward 21\n"
            "- Stronger-head trial retains marked doubling/halving asymmetry: "
            "insulin -6.027/+1.135; NE +5.102/-1.247 on the reported tuning sample.\n"
            "- Strength alone is not an established repair; its factual cost is still unmeasured.\n"
            "- Recovering architecture and calibration objects before choosing the next change.\n")
```