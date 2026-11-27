The probability check is accurate, but the new MAP directions are not yet validated repairs. I’ll recover the computation and retained fits before choosing a candidate. Success means stronger norepinephrine responses with factual NLL still within +0.005.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

r269_lines = Path("/work/notes.md").read_text().splitlines()
r269_hits = [
    i for i, s in enumerate(r269_lines)
    if any(t in s.lower() for t in
           ("identification", "instrument", "randomiz", "six-lag",
            "impulse", "record-estimated", "causal target"))
]
print("=== RECENT IDENTIFICATION NOTES ===")
for r269_i in r269_hits[-16:]:
    print("\n".join(r269_lines[max(0, r269_i-1):r269_i+4]))

r269_items = [(k, v) for k, v in list(globals().items())
              if not k.startswith(("_", "r269_"))]
print("=== RECENT RETAINED OBJECTS ===")
for r269_k, r269_v in r269_items[-110:]:
    if isinstance(r269_v, (np.ndarray, torch.Tensor)):
        print(r269_k, type(r269_v).__name__, tuple(r269_v.shape))
    elif inspect.isfunction(r269_v):
        print(r269_k, "function",
              r269_v.__code__.co_varnames[:r269_v.__code__.co_argcount])
    elif isinstance(r269_v, dict):
        print(r269_k, "dict", list(r269_v.keys())[:12])
    elif isinstance(r269_v, (tuple, list)):
        print(r269_k, type(r269_v).__name__, "length", len(r269_v))
    elif isinstance(r269_v, (str, int, float, np.number)):
        print(r269_k, repr(r269_v)[:160])
    else:
        print(r269_k, type(r269_v).__name__)

print("=== USER FUNCTIONS ===")
for r269_k, r269_v in r269_items:
    if inspect.isfunction(r269_v) and r269_v.__module__ == "__main__":
        print(r269_k, r269_v.__code__.co_varnames[:r269_v.__code__.co_argcount])
print("=== CURRENT EDITS ===", edited())

with open("/work/notes.md", "a") as r269_f:
    r269_f.write(
        "\n## Run 269: restart recovery\n"
        "- Reading architecture, identification evidence and retained interfaces.\n"
        "- Minimum-Fisher MAP routes remain locally tested only.\n"
        "- Submitted glucose-only strength 1.15 edit unchanged.\n"
        "- Python runs: 269; forward passes: 24.\n"
    )
```