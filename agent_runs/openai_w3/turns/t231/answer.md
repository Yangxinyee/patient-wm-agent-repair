The candidate removes the unwanted no-drug signal while barely changing factual tokens. That supports the gate fix, but factual accuracy and rollout effects remain untested. I’ll inspect the candidate and installation path before validating both.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED OBJECTS ===")
r208_keys = [
    k for k in globals()
    if not k.startswith("_") and any(
        s in k.lower() for s in [
            "193", "199", "206", "207", "checkpoint", "pilot",
            "candidate", "install", "restore", "restricted",
            "compens", "transport", "carrier", "prelude"
        ]
    )
]
for k in sorted(r208_keys):
    v = globals()[k]
    if isinstance(v, dict):
        print(k, "dict keys:", list(v.keys())[:35])
    elif callable(v):
        try:
            print(k, "callable", inspect.signature(v))
        except Exception:
            print(k, "callable")
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    else:
        print(k, type(v).__name__, repr(v)[:240])

print("\n=== CANDIDATE STATUS ===")
for k in ["candidate206", "checkpoint193", "restricted199"]:
    if k in globals() and isinstance(globals()[k], dict):
        for kk, vv in globals()[k].items():
            desc = (
                (type(vv).__name__, tuple(vv.shape))
                if hasattr(vv, "shape") else repr(vv)[:250]
            )
            print(k, kk, desc)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\nPython run 208; forwards 12. Restart recovery: candidate206 "
        "suppresses no-action signals, but is not installed. Inspecting "
        "architecture and retained candidate/checkpoint/install helpers "
        "before authoritative factual and rollout validation. "
        "No weights changed.\n"
    )
```