The helper source text is unavailable, and my notebook search missed the remaining causal section. No weights changed. I’ll inspect the cached arrays and helper metadata directly so the next experiment uses real dose changes rather than the misleading continuous probe.

```python
import re
import inspect

print("=== EARLY PERSISTENCE / TARGET NOTES ===")
p194 = notes191.find("## Run 6")
print(notes191[max(0, p194):max(0, p194) + 3600])

print("=== RUN-190 OBJECTS ===")
for n194, o194 in sorted(list(globals().items())):
    if not re.search(r"190$", n194):
        continue
    if isinstance(o194, dict):
        print(n194, "dict keys", list(o194.keys())[:20])
    elif hasattr(o194, "shape"):
        print(n194, type(o194).__name__, tuple(o194.shape))
    elif isinstance(o194, (str, int, float, list, tuple)):
        print(n194, repr(o194)[:300])

print("=== CACHED READING ARRAYS ===")
arr194 = []
for n194, o194 in list(globals().items()):
    sh194 = getattr(o194, "shape", ())
    if len(sh194) > 0 and sh194[0] in (4510, 4676):
        arr194.append((n194, type(o194).__name__, tuple(sh194)))
print(sorted(arr194)[:85])

print("=== META96 ===")
if "meta96" in globals():
    print(type(meta96), getattr(meta96, "shape", None))
    print(meta96.head(3).to_string() if hasattr(meta96, "head") else repr(meta96)[:1000])

print("=== CONTEXT / ACTION HELPERS ===")
for n194 in ("r124_context", "act96", "r124_lin", "r138_nll", "r144_nll"):
    o194 = globals().get(n194)
    if inspect.isfunction(o194):
        print(n194, inspect.signature(o194),
              "\nREFERENCES", o194.__code__.co_names,
              "\nCONSTANTS", repr(o194.__code__.co_consts)[:600])

print("=== FORWARD / STATE OBJECT NAMES ===")
print(sorted(n for n in globals() if re.search(
    r"(state|probe|atom|bundle|context|query|view|register|candidate|fallback)", n,
    flags=re.I))[:160])
```