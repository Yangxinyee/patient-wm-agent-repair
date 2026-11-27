The architecture confirms that the value head has 17 classes, including “no value,” while the probe used 16 centres. I’ll check how the vocabulary handles that class and recover the validation inputs. Correct bin handling should remove the crash; matching factual predictions remains a separate requirement.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== VALUE-BIN CONVENTIONS ===")
print("VOCAB top-level keys:", list(VOCAB))
print("codes:", VOCAB["codes"])
for key in ("value_bin_centres", "normalisation", "normalization", "config"):
    if key in VOCAB:
        print(key, repr(VOCAB[key])[:2300])
for event in ("glucose_finger", "map_arterial"):
    try:
        cc = centres153(event)
        print(event, "centres:", cc.shape, cc.tolist())
    except Exception as ex:
        print(event, type(ex).__name__, str(ex))

print("\n=== PROBE DEPENDENCIES ===")
for fn_name in ("readout153", "core_step136"):
    fn = globals().get(fn_name)
    if inspect.isfunction(fn):
        print(fn_name, "global names:", fn.__code__.co_names)
for name in sorted(list(globals())):
    obj = globals()[name]
    if inspect.isfunction(obj) and any(s in name.lower() for s in
              ("136", "137", "linear", "norm", "mlp", "attention", "transformer")):
        print(name, inspect.signature(obj))

print("\n=== RETAINED DATA AND PROBE OBJECTS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if name.endswith("153") or name in (
        "W", "weights", "states", "records", "train", "tuning", "metadata",
        "memory", "history", "out_new", "h_new", "context", "S"):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif hasattr(obj, "columns") and hasattr(obj, "shape"):
            print(name, "DataFrame", obj.shape, list(obj.columns))
        elif isinstance(obj, dict):
            print(name, "dict", len(obj), "keys:", list(obj)[:18])
        elif not inspect.isfunction(obj):
            print(name, type(obj).__name__, repr(obj)[:350])

print("\n=== FORWARD ARTIFACTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== DATA README ===")
print(Path("/data/README.md").read_text()[:3500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 155 — value-bin convention audit\n"
            "- Architecture identifies a 17-class value head (including absent) versus 16 physical centres.\n"
            "- Inspecting vocabulary and retained probe objects before repairing predicted-mean computation.\n"
            "- No weights changed.\n")
```