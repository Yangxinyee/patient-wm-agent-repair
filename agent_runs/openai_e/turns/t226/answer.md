The contrast arrays are available, but I used the wrong archive name for the saved direction. That error changed no weights. I’ll locate the correct archive and recover the earlier construction details, then compare new detectors against the existing one.

```python
import inspect
import re
import numpy as np
from pathlib import Path

print("ARCHIVE KEYS")
archives196 = {}
for p196 in sorted(Path("/work").glob("*.npz")):
    if p196.name.startswith(("ne_", "baseline", "insulin_candidate")):
        with np.load(p196, allow_pickle=False) as a196:
            archives196[p196.name] = {k: a196[k].copy() for k in a196.files}
            print(p196.name, [(k, a196[k].shape) for k in a196.files])

print("\nSINGLE-DIRECTION SOURCES")
for name196, a196 in archives196.items():
    if {"feature", "direction"}.issubset(a196):
        print(name196,
              "feature norm", float(np.linalg.norm(a196["feature"])),
              "direction norm", float(np.linalg.norm(a196["direction"])))
        for k196 in ["x2_slope", "choice"]:
            if k196 in a196:
                print(k196, a196[k196])

delta196 = archives196["ne_candidate151.npz"]["ne_delta"]
U196, S196, VT196 = np.linalg.svd(delta196, full_matrices=False)
print("\nCANDIDATE151")
print("gamma", archives196["ne_candidate151.npz"].get("gamma"))
print("top singular values", S196[:5])

print("\nRETAINED HELPERS")
for n196, v196 in list(globals().items()):
    if inspect.isfunction(v196) and re.search(
        r"prelude|posterior|detector|ridge|response|core|contrast|atom", n196, re.I
    ):
        c196 = v196.__code__
        print(n196, c196.co_varnames[:c196.co_argcount])

print("\nTARGETED CONSTRUCTION NOTES")
lines196 = Path("/work/notes.md").read_text().splitlines()
for run196 in [115, 147, 148, 149, 150, 151, 154, 155, 180, 191, 192]:
    matches196 = [
        j for j, line in enumerate(lines196)
        if re.match(rf"## Run {run196}(?:\D|$)", line)
    ]
    if matches196:
        j196 = matches196[-1]
        print("\n".join(lines196[j196:j196 + 3])[:460])

with Path("/work/notes.md").open("a") as f196:
    f196.write(
        "\n## Runs 195–196: archive lookup recovery\n"
        "Run195 confirmed paired train contrasts but stopped on a missing "
        "archive key. No weights changed. Located saved direction schemas and "
        "recovered prior NE construction notes. Python runs: 196; forwards: 22.\n"
    )
```