```python
import re, numpy as np, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD SCHEMA ===")
print(open("/data/README.md").read())
print("=== CURRENT EDITS ===", edited())

_g98 = list(globals().items())
print("=== RETAINED RECENT OBJECTS ===")
for _n98, _v98 in sorted(_g98):
    if _n98.startswith("_") or not re.search(r"(8[0-9]|9[0-7])$", _n98):
        continue
    if hasattr(_v98, "shape"):
        print(_n98, type(_v98).__name__, "shape", tuple(_v98.shape))
    elif isinstance(_v98, dict):
        print(_n98, "dict", list(_v98.keys())[:24])
    elif isinstance(_v98, (list, tuple)):
        print(_n98, type(_v98).__name__, "length", len(_v98))
    elif callable(_v98):
        print(_n98, "function")
    else:
        print(_n98, repr(_v98)[:350])

print("=== RETAINED FUNCTIONS ===")
for _n98, _v98 in sorted(_g98):
    if callable(_v98) and hasattr(_v98, "__code__") and not _n98.startswith("_"):
        print(_n98, _v98.__code__.co_varnames[:_v98.__code__.co_argcount])

print("=== LATENT / TRANSITION TENSORS ===")
for _n98 in NAMES:
    if any(_s98 in _n98.lower() for _s98 in ("z_in", "posterior", "post.", "transition")):
        _w98 = load(_n98)
        print(_n98, tuple(_w98.shape),
              "rms", float(np.sqrt(np.mean(np.asarray(_w98)**2))),
              "maxabs", float(np.max(np.abs(np.asarray(_w98)))))

with open("/work/notes.md", "a") as _f98:
    _f98.write("\n## Run 98\nRun 97 confirmed its proposed candidate was not executed and no edits exist. Recovering architecture and recent retained diagnostics to locate the nearly inactive latent pathway before selecting a repair; no weights changed.\n")
```
