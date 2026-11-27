The MAP addition fails the factual-fit limit: loss increased by 0.0091 nats, versus the allowed 0.005. I’ll recover the previously eligible glucose-only fallback and inspect the correction helpers. Any further MAP correction must improve dose responses while staying within that limit.

```python
from pathlib import Path
import numpy as np
import torch
import re
import types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== SAVED CANDIDATES ===")
for fn in ["recurrent_glucose_valid.npz", "recurrent_half_backup.npz"]:
    with np.load("/work/" + fn) as z:
        print(fn, [(k, z[k].shape, str(z[k].dtype)) for k in z.files])
        for k in z.files:
            if z[k].size <= 8:
                print(" ", k, z[k])

print("\n=== RECENT KERNEL OBJECTS ===")
_snapshot264 = list(globals().items())
for k, v in [
    (k, v) for k, v in _snapshot264
    if not k.startswith("_") and k not in {"NAMES", "SHAPES", "VOCAB"}
][-110:]:
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, types.FunctionType):
        print(k, "function", v.__code__.co_varnames[:v.__code__.co_argcount])
    elif isinstance(v, (list, tuple, dict)):
        print(k, type(v).__name__, "len", len(v))
    elif isinstance(v, (str, int, float, bool, np.number)):
        print(k, repr(v)[:160])

print("\n=== REPLAY / HEAD HELPERS ===")
for k, v in _snapshot264:
    if isinstance(v, types.FunctionType) and re.search(
        r"replay|predict|head|dose|route|rate|step|eval|embed", k, re.I
    ):
        print(k, v.__code__.co_varnames[:v.__code__.co_argcount],
              "uses:", v.__code__.co_names[:25])

print("\n=== IDENTIFICATION NOTES ===")
_evidence264 = [
    x for x in Path("/work/notes.md").read_text().splitlines()
    if re.search(r"instrument|randomiz|six.lag|impulse|identif|12\.37|4\.383|causal|confound", x, re.I)
]
print("\n".join(_evidence264[-45:]))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 264: recover after rejected MAP trial\n"
            "- Forward 22 factual delta +0.00909 exceeds the +0.005 limit.\n"
            "- Recovering the archived eligible glucose-only candidate and exact replay interfaces.\n"
            "- Current trial is not acceptable for submission; no weights changed.\n"
            "- Python runs: 264; forward passes: 22.\n")
```