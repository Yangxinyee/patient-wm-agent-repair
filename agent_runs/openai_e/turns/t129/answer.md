The recovered notes support a negative insulin effect and a positive norepinephrine effect, estimated from the records rather than medical assumptions. They also document weak drug-innovation recovery from the original model’s reading features. The output was truncated before the later evidence and saved-edit details, so I’ll retrieve those narrowly. This will show whether the next correction should target dose encoding, the readout, or both.

```python
import inspect, re
from pathlib import Path
import numpy as np
import torch

text111 = Path("/work/notes.md").read_text()
sections111 = re.split(r"(?=^## )", text111, flags=re.M)
print("=== SELECTED LATER RESULTS ===")
for section in sections111:
    heading = section.splitlines()[0] if section.splitlines() else ""
    match = re.search(r"Run (\d+)", heading)
    if match and int(match.group(1)) in [20, 25, 35, 45, 55, 58, 63, 70, 74, 75, 76, 77, 78]:
        print(section[:650])

print("\n=== EDIT ARCHIVES ===")
for filename in ["baseline77.npz", "ne_trial103.npz", "ne_reduced107.npz"]:
    with np.load("/work/" + filename, allow_pickle=False) as archive:
        print(filename, [(k, archive[k].shape) for k in archive.files])
        for k in archive.files:
            a = archive[k]
            if a.ndim == 0:
                print(k, a.item())

print("\n=== HEAD TENSORS ===")
for name in NAMES:
    if any(s in name for s in ["attr_heads.value", "monitor_head", "core.b_in",
                              "core.delta_in", "core.c_out"]):
        print(name, SHAPES[name])

snapshot111 = list(globals().items())
print("\n=== HELPERS ===")
for name, value in snapshot111:
    if inspect.isfunction(value) and any(
        s in name.lower() for s in ["core", "head", "posterior", "ne_fit", "dose", "causal"]
    ):
        try:
            print(name, str(inspect.signature(value))[:170])
        except (ValueError, TypeError):
            print(name)

print("\n=== EFFECT / POLICY VARIABLES ===")
for name, value in snapshot111:
    if name.startswith("_") or callable(value):
        continue
    if any(s in name.lower() for s in ["beta", "causal", "effect", "policy", "tau"]):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(name, tuple(value.shape))
            small = value.numel() <= 10 if isinstance(value, torch.Tensor) else value.size <= 10
            if small:
                print(value)
        elif isinstance(value, (int, float, str, dict)):
            print(name, repr(value)[:550])
        elif hasattr(value, "shape"):
            print(name, type(value).__name__, value.shape)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 111: focused recovery\n"
            "Early records-based estimates support negative insulin and positive NE effects, "
            "but later estimates and edit calibration must be recovered before revision. "
            "No weights changed. Python runs: 111; forwards: 15.\n")
```