The original-only surrogate was unreliable for MAP changes, but a later surrogate trained on paired official states achieved useful held-out accuracy for all four sensors. I’ll recover its routing results and see whether a candidate is ready for an official test. Its predicted dose shifts must survive that test; otherwise its validation does not generalize to the new directions.

```python
import re, numpy as np
from pathlib import Path

print("=== ROUTING RESULTS AFTER PAIRED-SURROGATE VALIDATION ===")
for s in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    if re.match(r"## Run[s]? (?:15[5-9]|16[0-9]|170)\b", s):
        print(s[:650])

print("\n=== PAIRED MODELS AND DERIVATIVE INTERFACE ===")
for k in ("models145", "diagnostics145", "readoutweights151", "dynamic_weights154",
          "events151", "indices151", "action_rows151", "targets151"):
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, dict):
        print(k, "keys", list(v)[:10])
        for a, b in list(v.items())[:4]:
            if isinstance(b, dict):
                print(" ", a, {c: (type(d).__name__, getattr(d, "shape", None)) for c, d in b.items()})
            elif isinstance(b, tuple):
                print(" ", a, [(type(c).__name__, getattr(c, "shape", None)) for c in b])
                if k == "models145":
                    print("  diagnostic", str(b[1])[:500])
            else:
                print(" ", a, type(b).__name__, getattr(b, "shape", None))
    elif isinstance(v, list):
        print(k, "len", len(v), str(v[:3])[:800])
    else:
        print(k, type(v).__name__, getattr(v, "shape", None))

print("\n=== RETAINED ROUTING OBJECTS (155–170) ===")
rows198 = []
for k, v in list(globals().items()):
    if re.search(r"(?:15[5-9]|16[0-9]|170)(?:$|_)", k) and not callable(v):
        if isinstance(v, dict):
            rows198.append((k, "dict", list(v)[:8]))
        elif hasattr(v, "shape"):
            rows198.append((k, type(v).__name__, tuple(v.shape)))
        elif isinstance(v, (str, int, float, np.number)):
            rows198.append((k, str(v)[:110]))
        elif isinstance(v, (list, tuple)):
            rows198.append((k, type(v).__name__, len(v)))
for r in rows198[:48]:
    print(r)

print("\n=== ROUTING HELPER SIGNATURES ===")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and re.search(r"(?:15[5-9]|16[0-9]|170)(?:$|_)", k):
        c = v.__code__
        print(k, "args", c.co_varnames[:c.co_argcount], "uses", c.co_names[:25])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 198: recover paired-surrogate routing results\n"
            "- Original-only surrogate failed edit-delta validation; paired surrogate achieved held-out delta R2 of about .72–.85.\n"
            "- Recovering its retained routing candidates before further official validation.\n"
            "- Python runs: 198; forward passes: 12. Saved edits unchanged.\n")
```
