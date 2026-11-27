The architecture confirms that doses enter a separate action encoder, while reading predictions also rely on last-value registers. That gives us a targeted editing route, but the recovered notebook was truncated before the key evidence. I’ll extract concise summaries and the available reconstruction functions before modifying anything.

```python
import re, inspect

sections = re.split(r"(?m)(?=^## )", notes_text)
print("=== NOTEBOOK SECTION DIGEST ===")
for section in sections:
    lines = section.strip().splitlines()
    if not lines:
        continue
    title = lines[0]
    if any(f"Run {i}" in title for i in range(31, 34)):
        continue
    prose = []
    for line in lines[1:]:
        s = line.strip()
        if s and not s.startswith(("{", "}", "[", "]", '"', "benchmark_stays=")):
            prose.append(s)
    digest = " ".join(prose)
    print(title)
    print(digest[:400])
    if len(digest) > 800:
        print("... END:", digest[-300:])

print("\n=== FUNCTIONS ===")
for key, obj in list(globals().items()):
    if not key.startswith("_") and inspect.isfunction(obj):
        try:
            print(key, inspect.signature(obj))
        except Exception:
            print(key)

print("\n=== NONTRIVIAL DATA OBJECTS ===")
for key, obj in list(globals().items()):
    if key.startswith("_") or key in {"obj"}:
        continue
    if hasattr(obj, "shape") and getattr(obj, "ndim", 0) >= 1:
        print(key, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict) and len(obj) > 0 and not key.startswith("__"):
        print(key, "dict", len(obj), "keys", str(list(obj)[:6])[:180])

print("\n=== VOCAB NORMALIZATION ===")
for key in VOCAB:
    if any(s in key.lower() for s in ["norm", "code", "config"]):
        print(key, str(VOCAB[key])[:2200])
```
