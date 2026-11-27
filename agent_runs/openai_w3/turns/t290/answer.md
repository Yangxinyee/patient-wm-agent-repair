The notebook’s record-based estimates support substantial insulin effects and positive norepinephrine effects; the original model largely missed them. The recovery output was truncated before the implementation details. I’ll retrieve a compact inventory next. This should reveal whether a joint correction is feasible without another poorly isolated weight change.

```python
import re, inspect

print("=== SELECTED NOTEBOOK ENTRIES ===")
for number in [180, 185, 190, 193, 194, 195, 196, 197, 200, 206, 220, 254, 255]:
    pat = rf"\nPython runs?:\s*{number}\b|\nPython run {number}\b"
    matches = list(re.finditer(pat, notes256))
    for m in matches[-1:]:
        end = notes256.find("\n\n", m.start() + 2)
        print(notes256[m.start():end if end >= 0 else m.start()+900][:650])

print("\n=== EXACT HELPER NAMES AND CACHE SHAPES ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    if (k.endswith(("193", "220", "254", "255")) or
        any(s in k.lower() for s in ["prelude", "replay", "checkpoint", "slotstep", "readout"])):
        v = globals()[k]
        if callable(v):
            try:
                sig = str(inspect.signature(v))
            except Exception:
                sig = "?"
            print(k, "FUNCTION", sig)
        elif isinstance(v, dict):
            print(k, "DICT", list(v)[:10])
        elif hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        else:
            print(k, type(v).__name__, str(v)[:100])

print("\n=== RECENT FUNCTIONS: AVAILABLE SOURCE ===")
for k in ["prelude254", "prelude255", "preludeE185"]:
    if k in globals() and callable(globals()[k]):
        try:
            print(k, inspect.getsource(globals()[k])[:1800])
        except Exception as exc:
            print(k, type(exc).__name__)
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 257; forwards 20. Run256 output truncated before implementation inventory. Recovered original rollout and record-derived references; requesting compact helper/cache inventory. No weights changed.\n")
```