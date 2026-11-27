```python
from pathlib import Path
import numpy as np, torch, json, re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
notes259 = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK ===")
print(notes259[:10500])
print("=== RETAINED HELPERS / REPLAY DATA ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_"):
        continue
    if callable(v) and hasattr(v, "__code__"):
        if any(s in k.lower() for s in ("replay", "install", "restore", "logit", "token", "latent", "head", "predict", "response", "prox", "route")):
            print(k, "args", v.__code__.co_varnames[:v.__code__.co_argcount],
                  "names", v.__code__.co_names)
    elif isinstance(v, (np.ndarray, torch.Tensor)) and any(s in k.lower() for s in ("216", "209", "230", "231", "dir", "resid", "effect", "action", "dose", "private")):
        print(k, tuple(v.shape), str(v.dtype))
print("=== SAVED CANDIDATE KEYS ===")
for f in ("neutral209.npz", "token_directions215.npz", "response_candidates216.npz",
          "response_directions231.npz", "response_fd230.npz", "fallback188.npz"):
    z = np.load("/work/" + f)
    print(f, [(k, z[k].shape) for k in z.files])
print("=== LATEST FORWARD FILES ===")
for d in sorted(Path("/work").glob("fwd_*"), key=lambda x: int(x.name.split("_")[-1]))[-5:]:
    print(d.name, [p.name for p in d.iterdir()])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 259 — restart recovery\n"
            "- token257 remains rejected; neutral209 is installed.\n"
            "- Attention has no KV bias, so the bias-leakage hypothesis is refuted.\n"
            "- Recovering architecture, record-based targets, and replay helpers before a targeted edit.\n")
```
