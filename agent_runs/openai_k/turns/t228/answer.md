The diagnostic found variation in the reserved dose channels, but failed when it selected an empty subset. That is a selection error, not evidence about drug effects. I’ll recover the architecture and available replay helpers before changing weights; the next useful test must validate drug-specific responses against official predictions, not just local head estimates.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== CURRENT EDITS ===")
for name in edited():
    a = load(name)
    b = load_original(name)
    aa = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    bb = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
    print(name, aa.shape, "delta_norm", float(np.linalg.norm(aa-bb)))

snapshot207 = dict(globals())
print("\n=== AVAILABLE FUNCTIONS ===")
for name, obj in snapshot207.items():
    if inspect.isfunction(obj) and not name.startswith("_"):
        code = getattr(obj, "__code__", None)
        print(name, list(code.co_varnames[:code.co_argcount]) if code else "")

print("\n=== RELEVANT RETAINED OBJECTS ===")
tokens207 = ("audit", "probe", "replay", "candidate", "baseline", "contexts",
             "action", "surrogate", "best188", "r132", "mask", "W139")
for name, obj in snapshot207.items():
    if not any(t.lower() in name.lower() for t in tokens207):
        continue
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys", list(obj)[:12])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), "sample", repr(obj[:3])[:180])
    elif hasattr(obj, "columns"):
        print(name, type(obj).__name__, obj.shape, "columns", list(obj.columns))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 207: recover authoritative replay interface\n"
            "- Run 206 measured nonzero reserved-channel variation, then failed on an empty selection; this was a diagnostic indexing failure.\n"
            "- Reading architecture and schema and locating retained replay functions before further edits.\n"
            "- No weights changed. Python runs: 207; forward passes: 14.\n")
```