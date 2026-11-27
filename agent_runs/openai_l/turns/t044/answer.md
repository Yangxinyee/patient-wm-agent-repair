The posterior reconstructions still have substantial error, so I would not use them to calculate an edit yet. I’ll check the exact action encoding, state timing, and retained functions. A correct reconstruction should match the exported posterior means nearly exactly; persistent error would refute the current encoding assumptions.

```python
import os, json, inspect, numpy as np

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

print("\n=== RETAINED FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig)

print("\n=== RELEVANT RETAINED OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if any(s in name.lower() for s in ("action", "poster", "train_ids", "r32_best", "kernel", "wide", "panel")):
        if not inspect.isfunction(obj):
            shape = getattr(obj, "shape", None)
            if isinstance(obj, dict):
                print(name, type(obj).__name__, "keys:", list(obj)[:20])
            elif shape is not None:
                print(name, type(obj).__name__, shape)
            elif isinstance(obj, (list, tuple)) and len(obj) < 20:
                print(name, repr(obj))
            else:
                print(name, type(obj).__name__)

print("\n=== FORWARD 4 FILES AND ARRAY SCHEMAS ===")
for filename in sorted(os.listdir("/work/fwd_4")):
    path = "/work/fwd_4/" + filename
    print(filename)
    if filename.endswith(".npz"):
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                a = arrays[key]
                print(" ", key, a.shape, a.dtype,
                      a[:3].tolist() if a.ndim == 1 else "")

print("\n=== ACTION/POSTERIOR TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in ("action", "act_", "z_post", "bundle", "prelude")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 39\nNo edits. Run 38 posterior reconstruction errors remain material "
            "(insulin 0.076, NE 0.119). Recovering exact encoding, state timing, "
            "retained functions, and forward-4 schema before calculating an edit.\n")
```