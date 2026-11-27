The current helper uses value column 2 and presence column 3, while other layouts fit the GPU means better. That could make a decoder look harmless locally but harmful on the real model. I will recover the stored head edits and matching artifacts. Their paired GPU changes should let us test this explanation directly.

```python
import numpy as np
import torch
from pathlib import Path

print("=== RECONSTRUCTION CONSTANTS AND RETURN ===")
print(reconstruct_reading_head.__code__.co_consts)
print("return type:", type(result100).__name__)
if isinstance(result100, (tuple, list)):
    print("return shapes:",
          [tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
           for x in result100])

print("\n=== STORED SMALL NUMERIC EDIT ARRAYS ===")
for name, value in list(globals().items()):
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        continue
    if isinstance(value, torch.Tensor):
        if not value.dtype.is_floating_point:
            continue
        array = value.detach().cpu().numpy()
    else:
        if not np.issubdtype(value.dtype, np.number):
            continue
        array = value
    if array.shape in [(192,), (17,), (17, 192), (192, 17), (193,)]:
        if any(s in name.lower() for s in [
            "weight", "bias", "coef", "delta", "candidate", "repair",
            "tilt", "raw", "probe", "w80", "b80", "w87", "b87"
        ]):
            print(name, array.shape,
                  "norm:", round(float(np.linalg.norm(array)), 6))
            if array.shape == (17, 192):
                base102 = np.asarray(load_original("attr_heads.value_bin.2.weight"))
                singular102 = np.linalg.svd(array - base102, compute_uv=False)
                print("delta singular values:", singular102[:4])

print("\n=== STORED REGRESSION OBJECTS ===")
for name, value in list(globals().items()):
    if hasattr(value, "coef_"):
        print(name, type(value).__name__, np.shape(value.coef_))
    elif isinstance(value, dict) and any(s in name.lower() for s in [
        "head", "repair", "candidate", "tilt"
    ]):
        print(name, "keys:", list(value.keys())[:8])

print("\n=== FORWARD ARTIFACTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== VOCAB NORMALISATION EXAMPLE ===")
print("config:", VOCAB["config"])
print("normalisation-related keys:",
      [k for k in VOCAB if any(s in k for s in ["norm", "median", "iqr"])])
print("repair_norms glucose_finger:", repair_norms.get("glucose_finger"))

print("\n=== REJECTED GPU EDIT ===")
print(cmp94.groupby("event")[["dnll", "dmean"]].mean().to_string())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Runs 96–102: restart audit\n"
        "- Record estimates support immediate insulin/glucose and NE/MAP effects; "
        "insulin persistence remains weakly identified.\n"
        "- last_numeric has eight inputs; current reconstruction uses value2/presence3, "
        "but alternative layouts fit GPU means better.\n"
        "- Recovering paired rejected-edit outputs as a reconstruction fingerprint.\n"
        "- No edits; 12 forward passes used.\n"
    )
```