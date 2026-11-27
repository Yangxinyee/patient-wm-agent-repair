The architecture confirms that the current edit changes recurrent updates, so leakage can affect later readings as well as the immediate prediction. The output was truncated before the retained helpers appeared. I’ll inspect those helpers and the accepted insulin edit compactly. That will show whether a better-centered NE correction is feasible without changing the validated insulin baseline.

```python
import inspect, numpy as np, torch
from pathlib import Path

print("KEY INSULIN VALIDATION NOTES")
for section in Path("/work/notes.md").read_text().split("\n## "):
    title = section.splitlines()[0]
    if any(f"Run {j}:" in title for j in [58, 63, 65, 67]):
        print(("## " + section)[:950])

print("\nHELPERS")
for name, obj in sorted(list(globals().items())):
    if callable(obj) and hasattr(obj, "__code__"):
        if any(s in name.lower() for s in [
            "core", "coda", "head", "prelude", "contrast", "local",
            "action", "update", "sensitivity", "fit_detector"
        ]):
            print(name, str(inspect.signature(obj)),
                  "refs=" + ",".join(obj.__code__.co_names)[:210])

print("\nRETAINED NE / CONSTRAINT OBJECTS")
for name, obj in sorted(list(globals().items())):
    if any(s in name.lower() for s in [
        "81", "82", "83", "77", "constraint", "null", "design", "prelude"
    ]):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys", list(obj)[:14])
        elif isinstance(obj, (int, float, str, tuple)):
            print(name, str(obj)[:180])

print("\nSAVED ARRAYS")
for path in sorted(Path("/work").glob("*.npz")):
    with np.load(path, allow_pickle=False) as z:
        print(path.name, {k: z[k].shape for k in z.files})

print("\nCORE AND VALUE TENSORS")
for name in NAMES:
    if any(s in name for s in [
        "core.b_in", "core.delta_in", "core.c_out", "z_post",
        "attr_heads.value_bin"
    ]):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 91: compact helper recovery\n"
        "Architecture recovered; run90 output truncated before helpers. "
        "Inspecting retained NE fits and validated insulin edit without another "
        "large architecture dump. No weights changed. "
        "Python runs: 91; forwards: 12.\n"
    )
```