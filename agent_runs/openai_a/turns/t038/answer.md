The CPU head’s errors depend strongly on the last reading, and latent averaging did not fix them. I suspect an input or normalization mismatch. I’ll recover the architecture and retained variables before editing; agreement with the forward tool would confirm the reconstruction, while continued systematic errors would refute it.

```python
import os, json, inspect, types, numpy as np, pandas as pd, torch
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA COLUMNS ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:-8000][-14000:] if len(notes_text) > 8000 else notes_text)
print("\n=== RETAINED VARIABLES ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns)[:24])
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict) and len(obj) < 60:
        print(name, "dict keys", list(obj)[:30])
    elif isinstance(obj, types.FunctionType):
        print(name, "function", str(inspect.signature(obj)))
print("\n=== VOCAB STRUCTURE ===")
for key, val in VOCAB.items():
    if key not in {"value_bin_edges", "value_bin_centres", "rate_bin_edges", "rate_bin_centres"}:
        print(key, val)
for skill_path in [
    "/Users/<user>/.codex/skills/evaluate-engineering-decisions/SKILL.md"
]:
    if os.path.isfile(skill_path):
        print("\n=== ENGINEERING DECISION SKILL ===")
        print(Path(skill_path).read_text())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart after Python 32\n"
            "- No edits; forward 4/50. Head mismatch depends on last-value range and is not repaired by latent averaging.\n"
            "- Recovering architecture, retained variables, and earlier conclusions before choosing a weight edit.\n")
```
