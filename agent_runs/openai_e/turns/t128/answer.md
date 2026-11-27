The inspection recovered the model-state and readout data, but stopped before testing a new correction; the weights are unchanged. The reduced norepinephrine edit passed the tested factual constraint but produced only a small intervention response. Next I’ll recover the recorded-effect estimates and architecture before choosing a different edit location. A useful alternative must increase dose sensitivity without shifting factual predictions.

```python
from pathlib import Path
import inspect, re
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes110 = Path("/work/notes.md").read_text()
print("\n=== EARLY EVIDENCE ===")
print(notes110[:8000])
for marker in ["## Run 58", "## Run 63", "## Run 77", "## Run 78"]:
    pos = notes110.find(marker)
    if pos >= 0:
        end = notes110.find("\n## ", pos + 4)
        print(notes110[pos:end if end >= 0 else pos + 2500][:2500])

print("\n=== SAVED EDITS ===")
for filename in ["baseline77.npz", "insulin_candidate63.npz",
                 "ne_trial103.npz", "ne_reduced107.npz"]:
    with np.load("/work/" + filename, allow_pickle=False) as archive:
        print(filename)
        for key in archive.files:
            value = archive[key]
            print(" ", key, value.shape,
                  value.item() if value.ndim == 0 else
                  ("norm=" + str(float(np.linalg.norm(value)))
                   if np.issubdtype(value.dtype, np.number) else str(value)[:150]))

print("\n=== RETAINED COMPUTATION HELPERS ===")
snapshot110 = list(globals().items())
for name, value in snapshot110:
    if inspect.isfunction(value) and any(
        token in name.lower()
        for token in ["core", "head", "posterior", "dose", "causal", "jac", "ne_fit"]
    ):
        try:
            print(name, inspect.signature(value))
        except (TypeError, ValueError):
            print(name)

print("\n=== RETAINED EFFECT / POLICY OBJECTS ===")
for name, value in snapshot110:
    if name.startswith("_") or callable(value):
        continue
    if any(token in name.lower()
           for token in ["causal", "beta", "effect", "target", "policy", "panel", "tau"]):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(name, type(value).__name__, tuple(value.shape))
            if value.numel() <= 12 if isinstance(value, torch.Tensor) else value.size <= 12:
                print(value)
        elif isinstance(value, (int, float, str, dict, tuple, list)):
            print(name, repr(value)[:1200])
        elif hasattr(value, "shape"):
            print(name, type(value).__name__, value.shape,
                  list(value.columns)[:25] if hasattr(value, "columns") else "")

print("\nCURRENT EDITS", edited())
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 110: restart recovery before redesign\n"
        "Run109 stopped during inspection; no new weights were installed. "
        "Recovering architecture, record-derived effects, saved edit composition, "
        "and local helpers. Reduced NE correction remains installed; its tested "
        "factual NLL increase was .00402, leaving little safety margin. "
        "Python runs: 110; forwards: 15.\n"
    )
```