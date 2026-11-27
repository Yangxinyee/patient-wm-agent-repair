The architecture confirms that the edit changes reading distributions without changing the recurrent dynamics. The notebook supports an insulin effect, but the norepinephrine estimates remain unresolved. I’ll recover the exact graft helpers and identification checks; this will determine whether factual recalibration is justified or whether the current checkpoint is safer.

```python
import re, inspect, numpy as np

print("=== FUNCTION NAMES ===")
print(", ".join(sorted(n for n, x in list(globals().items())
                       if not n.startswith("_") and inspect.isfunction(x))))

print("\n=== CHECKPOINTS AND CACHES ===")
for n, x in list(globals().items()):
    if n.startswith("_"):
        continue
    if any(s in n.lower() for s in ("checkpoint", "cache", "gate", "activity", "r3", "sc_rank")):
        if isinstance(x, dict):
            print(n, "dict", [(str(k), getattr(v, "shape", type(v).__name__))
                             for k, v in list(x.items())[:10]])
        elif hasattr(x, "shape"):
            print(n, type(x).__name__, x.shape)
        elif isinstance(x, (int, float, str, bool)):
            print(n, repr(x)[:120])
        elif isinstance(x, (list, tuple)):
            print(n, type(x).__name__, "length", len(x))

print("\n=== IDENTIFICATION NOTEBOOK EXCERPTS ===")
sections = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
chosen = []
for sec in sections:
    title = sec.splitlines()[0] if sec.splitlines() else ""
    if any(s in title.lower() for s in
           ("identif", "instrument", "causal", "discontinu", "boundary", "persistence", "negative control")):
        chosen.append(sec)
for sec in chosen[:6]:
    print(sec[:850])
for sec in chosen[-3:]:
    if sec not in chosen[:6]:
        print(sec[:850])
```
