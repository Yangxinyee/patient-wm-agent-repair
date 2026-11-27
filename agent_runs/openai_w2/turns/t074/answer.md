The architecture confirms that drug inputs have a separate encoder, while predictions also depend directly on the last-reading register. The notebook output was truncated, so I’ll recover the relevant diagnostics before choosing an edit. This should show whether weak dose responses come from action processing or from how predictions carry effects forward.

```python
import re
notes_text = open("/work/notes.md").read()
print("NOTEBOOK LENGTH:", len(notes_text))
print("NOTEBOOK HEADINGS:")
for m in re.finditer(r"^## .*", notes_text, re.M):
    print(m.start(), m.group())
print("\nKERNEL ARRAYS/DATAFRAMES:")
items = list(globals().items())
for k, v in sorted(items):
    if not k.startswith("_") and hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape)
print("\nKERNEL CONTAINERS:")
for k, v in sorted(items):
    if not k.startswith("_") and isinstance(v, (dict, list, tuple)) and k not in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        print(k, type(v).__name__, len(v))
print("\nFUNCTION SIGNATURES:")
for k, v in sorted(items):
    if not k.startswith("_") and callable(v) and hasattr(v, "__code__"):
        c = v.__code__
        print(k, c.co_varnames[:c.co_argcount])
print("\nRELEVANT TENSOR SHAPES:")
for n in NAMES:
    if any(s in n for s in ("prelude", "attr_heads.value", "last_numeric", "embed_transition", "core.flow", "core.a_in", "core.delta_in", "z_prior", "z_post")):
        print(n, SHAPES[n])
print("\nFORWARD FILES:")
import os
for i in range(1, 13):
    p = f"/work/fwd_{i}"
    print(i, os.listdir(p) if os.path.exists(p) else "missing")
```
