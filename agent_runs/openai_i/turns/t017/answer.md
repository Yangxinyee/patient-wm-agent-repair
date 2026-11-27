The tensor shapes expose a reconstruction issue: `last_numeric` takes eight inputs, while the architecture describes them only schematically. I’ll recover the existing helper definitions and compact vocabulary metadata. If correcting those inputs removes the prediction mismatch, we can evaluate candidate head edits locally rather than spending forward passes blindly.

```python
import inspect, json
import numpy as np
from pathlib import Path

print("=== FUNCTION NAMES ===")
for nm, ob in list(globals().items()):
    if inspect.isfunction(ob) and nm not in {
        "load", "load_original", "save_edit", "drop_edit", "edited", "compact"
    }:
        print(nm, inspect.signature(ob), "names:", ob.__code__.co_names[:18])

print("\n=== LAST 65 GLOBAL OBJECTS ===")
for nm in list(globals())[-65:]:
    ob = globals()[nm]
    if isinstance(ob, np.ndarray):
        print(nm, "array", ob.shape, ob.dtype)
    elif hasattr(ob, "columns") and hasattr(ob, "shape"):
        print(nm, "table", ob.shape, list(ob.columns))
    elif isinstance(ob, dict):
        print(nm, "dict", list(ob)[:12])
    elif isinstance(ob, (list, tuple)):
        print(nm, "list", len(ob), str(ob)[:100])
    elif isinstance(ob, (str, int, float, bool)):
        print(nm, repr(ob)[:150])
    elif hasattr(ob, "files"):
        print(nm, "npz", ob.files)
    elif inspect.isfunction(ob):
        print(nm, "function")

print("\n=== FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== VOCAB KEYS AND CONFIG ===")
print(list(VOCAB))
print(json.dumps(VOCAB.get("config", {})))
print("codes:", json.dumps(VOCAB.get("codes", {})))
for key in VOCAB:
    if any(s in key.lower() for s in ("norm", "scal", "categor", "unit")):
        print(key, str(VOCAB[key])[:1500])

print("\n=== ADDITIONAL HEAD TENSOR NAMES ===")
for nm in NAMES:
    if any(s in nm for s in ("gap.", "time_head", "filter_", "pending_", "last_", "monitor")):
        print(nm, SHAPES[nm])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Reconstruction inspection — Python 12; forward 4\n"
        "- Architecture recovered. last_numeric.weight is (192,8); "
        "the four-field schematic is insufficient to reproduce its inputs.\n"
        "- No edits; investigating register features, time flow and latent marginalization.\n"
    )
```
