The local reconstruction is close, but its loss is not reliable enough to certify an edit. The original dose responses also look weak relative to the provisional estimates from the records. I’ll inspect the update equations before choosing a reversible edit; it must improve dose responses while preserving factual fit.

```python
from pathlib import Path
import json

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, tuple(SHAPES[name]))

print("\n=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    obj = globals()[name]
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print("function:", name)
    elif name in {
        "panel", "HC", "head_rows", "state_x", "head_weights",
        "impulse", "impulses", "dose_models", "train", "tuning",
        "train_df", "tuning_df", "W", "weights"
    }:
        print(name, type(obj).__name__, getattr(obj, "shape", ""))

print("\n=== CURRENT EDITS ===")
print(edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Restart — Python 24; forward 4\n"
        "- No weights changed. Local head reconstruction is approximate and cannot certify factual loss.\n"
        "- Inspecting full architecture and tensor shapes to design a reversible dose-sensitive edit.\n"
        "- Record-based impulse estimates remain provisional; factual certification will use the forward tool.\n"
    )
```