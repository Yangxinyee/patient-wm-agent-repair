The architecture confirms that the value head combines shared state with sensor-specific history, which may support a selective correction. The output was truncated before the cached fits appeared. I’ll inspect those objects compactly; a useful correction must distinguish sensor families and remain quiet on held-out factual states.

```python
import re
print("=== VALUE HEAD / REGISTER TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ("attr_heads.value_bin", "last_numeric", "query_norm",
                             "embed_transition.value_bin")):
        print(n, SHAPES[n])

print("\n=== RECENT OBJECT NAMES ===")
recent202 = sorted(k for k in list(globals())
                   if re.search(r"(179|18[0-9]|19[0-9]|200)$", k))
print(" ".join(recent202))

print("\n=== RECENT DICTIONARIES ===")
for k in recent202:
    v = globals()[k]
    if isinstance(v, dict):
        desc = []
        for a, b in list(v.items())[:8]:
            if isinstance(b, dict):
                desc.append((str(a), "dict:" + ",".join(map(str, list(b)[:12]))))
            else:
                desc.append((str(a), tuple(b.shape) if hasattr(b, "shape")
                             else type(b).__name__))
        print(k, desc)
    elif callable(v):
        print(k, "function", str(inspect.signature(v)))

print("\n=== KEY ARRAYS ===")
for k in recent202:
    v = globals()[k]
    if hasattr(v, "shape") and not isinstance(v, dict):
        print(k, tuple(v.shape))

print("\n=== EMPIRICAL KERNEL NOTES ===")
text202 = Path("/work/notes.md").read_text()
for marker in ("## Run 19", "## Run 20", "## Run 21"):
    pos = text202.find(marker)
    if pos >= 0:
        print(text202[pos:pos+1700])
print("\nCurrent edits:", edited())
```