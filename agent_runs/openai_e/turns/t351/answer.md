The architecture confirms that dose edits affect recurrent states and subsequent sampled readings, so local response estimates alone cannot validate an edit. The output was truncated before the archive details. I’ll retrieve those details compactly and check the installed weights; an exact match to candidate B would confirm that the reported results belong to the intended edit.

```python
from pathlib import Path
import numpy as np
import re

print("=== ARCHIVE CONTRACTS ===")
for fn in ["joint236.npz", "joint274.npz", "joint283.npz", "joint284.npz"]:
    with np.load("/work/" + fn, allow_pickle=False) as z:
        print(fn)
        for k in z.files:
            a = z[k]
            print(" ", k, a.shape, str(a.dtype),
                  a.tolist() if a.size <= 8 else
                  {"norm": float(np.linalg.norm(a)), "min": float(a.min()), "max": float(a.max())})

print("\n=== INSTALLED WEIGHT ===")
w0_290 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
wc_290 = np.asarray(load("core.b_in.weight"), dtype=np.float32)
print("shape", wc_290.shape, "delta_norm", float(np.linalg.norm(wc_290-w0_290)))
print("edits", edited())

print("\n=== REUSABLE OBJECT NAMES ===")
objs_290 = [(k, type(v).__name__, getattr(v, "shape", None))
            for k, v in list(globals().items())
            if not k.startswith("_") and
            re.search(r"target|causal|beta|delta|direction|component|coef|basis|joint|blend|scale", k, re.I)]
print(objs_290[:65])

notes_290 = Path("/work/notes.md").read_text()
sections_290 = re.split(r"(?=^## )", notes_290, flags=re.M)
print("\n=== IDENTIFICATION / CONSTRUCTION AUDIT ===")
selected_290 = []
for s in sections_290:
    heading = s.splitlines()[0] if s.splitlines() else ""
    if (re.search(r"Run[s]? (?:6|70|148|151|154|155|199|219|236|250|255|262|264|265|266)\b", heading)
        or ("identif" in heading.lower() and not re.search(r"Run[s]? 28", heading))):
        selected_290.append(s.strip())
audit_290 = "\n\n".join(selected_290)
print(audit_290[:4700])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 290: bounded archive and identification audit\n"
        "Run289 output truncated before archive contracts. Retrieved compact "
        "candidate metadata, installed recurrent-weight delta, reusable kernel "
        "objects and selected identification/construction notes. No weights changed.\n"
        "Python runs: 290; forwards: 47.\n"
    )
```