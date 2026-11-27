The adjusted insulin estimate is stable near −2 per rate unit, and norepinephrine is consistently positive near +3.3. The insulin estimates also suggest effects persist beyond one hour, though the long-lag estimates remain tentative. I’ll recover the remaining persistence evidence and correction helpers; unstable persistence would argue against calibrating to a precise rollout target.

```python
import inspect

s193 = notes191.find('    "norepinephrine": {', notes191.find("## Run 5"))
print("=== REMAINING EARLY CAUSAL NOTES ===")
print(notes191[s193:s193 + 3900])

print("=== HELPER DEFINITIONS ===")
for n193 in ("r129_queries", "r138_pair", "r159_pair", "rh_eval", "mean190"):
    o193 = globals().get(n193)
    if not inspect.isfunction(o193):
        continue
    print("\nFUNCTION", n193, inspect.signature(o193))
    try:
        src193 = inspect.getsource(o193)
        print(src193[:2200])
    except Exception:
        print("REFERENCES", o193.__code__.co_names)
        print("LOCALS", o193.__code__.co_varnames)
        print("CONSTANTS", repr(o193.__code__.co_consts)[:1000])

print("=== MODEL / PROBE CLASSES ===")
print([n for n, o in list(globals().items())
       if inspect.isclass(o) and getattr(o, "__module__", "") in ("__main__", "builtins")])

print("=== RECENT VARIABLE NAMES ===")
print(sorted(n for n in globals()
             if n.startswith(("c176", "c178", "c182", "c186", "c187",
                              "c188", "c189", "c190", "r178", "r182"))))
```