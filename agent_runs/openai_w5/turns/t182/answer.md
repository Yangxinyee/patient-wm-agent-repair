The earlier analysis found record-derived one-step effects and persistence, but those rollout targets still depend on identification assumptions. The helper listing also expanded large defaults and was truncated. I’ll retrieve only compact metadata so I can trace the exposure channel without changing weights.

```python
import re

print("=== LOCAL FUNCTION ARGUMENTS (NO DEFAULT VALUES) ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if inspect.isfunction(o) and getattr(o, "__module__", "") == "__main__":
        c = o.__code__
        print(n, c.co_varnames[:c.co_argcount])

print("=== RECENT OBJECT METADATA ===")
for n in sorted(list(globals())):
    if re.search(r"(14[4-9]|15[0-5])(?:_|$)", n):
        o = globals()[n]
        if isinstance(o, dict):
            print(n, "dict", list(o.keys())[:16])
        elif isinstance(o, pd.DataFrame):
            print(n, "df", o.shape, list(o.columns)[:12])
        elif hasattr(o, "shape"):
            print(n, type(o).__name__, tuple(o.shape))
        elif isinstance(o, (int, float, str)):
            print(n, repr(o)[:100])
        elif isinstance(o, (list, tuple)):
            print(n, type(o).__name__, "length", len(o))
        elif not inspect.ismodule(o) and not inspect.isfunction(o):
            print(n, type(o).__name__)

print("=== SPECS ===")
for s in _r148_specs:
    print({k: s[k] for k in ["event", "drug", "beta", "c"] if k in s},
          "keys", list(s.keys()))
print("=== CACHE / TRANSPORT OBJECT NAMES ===")
print([n for n in globals() if any(
    k in n.lower() for k in ["clean", "channel", "transport", "rollout", "cache", "ids"]
) and not inspect.isfunction(globals()[n])])
print("=== RECENT FORWARD FILES ===")
for i in range(13, 19):
    d = Path(f"/work/fwd_{i}")
    if d.exists():
        print(d.name, [p.name for p in d.iterdir()])
```
