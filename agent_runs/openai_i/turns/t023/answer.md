The architecture confirms that `last_numeric` has eight inputs, although its description names only four register fields. That mismatch could explain why rows with previous readings reconstruct poorly. I’ll inspect the retained functions and tensor shapes before testing the missing fields; a correct layout should improve both predicted means and NLL.

```python
import dis
from pathlib import Path
import numpy as np

print("=== VOCABULARY AND NORMALIZATION ===")
print("config:", VOCAB["config"])
print("codes:", VOCAB["codes"])
print("event_kinds:", VOCAB["event_kinds"])
print("atom_kinds:", VOCAB["atom_kinds"])
print("numeric:", VOCAB["numeric"])

print("\n=== TENSOR INVENTORY ===")
for name in NAMES:
    print(name, SHAPES[name])

for name in ["reconstruct_head", "head2", "_fn"]:
    obj = globals().get(name)
    if callable(obj) and hasattr(obj, "__code__"):
        print("\n=== FUNCTION", name, "===")
        print("defaults:", obj.__defaults__)
        print("constants:", obj.__code__.co_consts)
        print("names:", obj.__code__.co_names)
        dis.dis(obj)

print("\n=== RETAINED ANALYSIS OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in ["train", "tuning", "df", "state_x", "head_rows"]:
        continue
    if hasattr(obj, "shape") and not callable(obj):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict) and name not in ["VOCAB", "SHAPES", "head_weights", "__builtins__"]:
        print(name, "dict keys:", list(obj)[:15])

print("\n=== SAVED FORWARD FILES ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    for path in sorted(directory.iterdir()):
        print(path, path.stat().st_size)

print("\n=== LAST-NUMERIC COLUMN MAGNITUDES ===")
W = load_original("last_numeric.weight")
print("column norms:", np.linalg.norm(W, axis=0))
print("column means:", W.mean(axis=0))

with open("/work/notes.md", "a") as f:
    f.write("\n## Exact input investigation — Python 18; forward 4\n"
            "- Architecture describes four register fields but last_numeric has eight columns.\n"
            "- Inspecting function bytecode, vocabulary and tensor inventory to resolve the mismatch.\n"
            "- No weights changed.\n")
```