The records support a substantial insulin effect and a smaller positive norepinephrine effect, while the original rollout misses both. The architecture also confirms that actions can affect the recurrent state directly. I’ll recover the retained calibration helpers without the long notebook output; an exact replay would let us use them, otherwise I’ll calibrate with forward passes.

```python
import inspect, json, re
g122 = dict(globals())
print("=== FUNCTION INVENTORY ===")
for k, v in sorted(g122.items()):
    if inspect.isfunction(v) and not k.startswith("_"):
        try:
            s = str(inspect.signature(v))
        except Exception:
            s = "?"
        print(k, s)
print("=== RECENT OBJECTS (100-121) ===")
lines123 = []
for k, v in sorted(g122.items()):
    m = re.match(r"r(\d+)", k)
    if m and 100 <= int(m.group(1)) <= 121:
        desc = type(v).__name__
        if hasattr(v, "shape"):
            desc += " " + str(tuple(v.shape))
        elif isinstance(v, dict):
            desc += " keys=" + str(list(v)[:8])
        elif isinstance(v, (list, tuple, str)):
            desc += " len=" + str(len(v))
        elif isinstance(v, (int, float, bool)):
            desc += " " + repr(v)
        lines123.append(k + ": " + desc)
print("\n".join(lines123)[:6000])
print("=== NON-r DATA OBJECTS ===")
print([(k, type(v).__name__, getattr(v, "shape", None)) for k, v in g122.items()
       if not k.startswith("r") and hasattr(v, "shape")][:45])
print("=== TENSOR GROUPS ===")
print([(n, SHAPES[n]) for n in NAMES if any(s in n for s in
       ("z_post", "z_head", "prelude.0", "attr_heads.value", "last_", "type_emb", "query_norm"))])
```