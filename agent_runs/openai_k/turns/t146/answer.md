The decoder mostly tracks physical-dose changes, but its factual target is a dose innovation—not the dose itself. The prior-feature test failed because it did not capture the required context. I’ll recover the recurrent interface before testing drug-specific edits. Success requires better dose responses without exceeding the factual-loss limit.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA COLUMNS ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
for k, v in list(globals().items()):
    if inspect.isfunction(v) and not k.startswith("_"):
        print(k, inspect.signature(v), "references:", v.__code__.co_names[:35])

print("\n=== RETAINED CANDIDATE ARRAYS ===")
for k, v in list(globals().items()):
    if any(s in k.lower() for s in
           ["66", "74", "88", "90", "candidate", "impulse", "target", "prior", "decoder"]):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape), str(v.dtype))
        elif isinstance(v, dict):
            print(k, "dict keys:", list(v.keys())[:18])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "length:", len(v))

print("\n=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 135: recover recurrent calibration interface\n"
            "- Decoder intervention differences mostly track physical-dose changes.\n"
            "- Prior-feature assay failed because context was not captured.\n"
            "- Recovering interfaces before official drug-separated calibration.\n"
            "- No edits. Python runs: 135; forward passes: 6.\n")
```