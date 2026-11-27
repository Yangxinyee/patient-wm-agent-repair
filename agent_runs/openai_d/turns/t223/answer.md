The architecture confirms that dose edits change both numeric scores and discrete bins, whereas the recovered probe changed only a continuous embedding direction. The records also show substantial treatment-selection confounding. I’ll recover the adjusted-effect estimates and existing forward helpers; consistent estimates across controls would support the targets, while instability would weaken them.

```python
import inspect

start192 = notes191.find("## Run 4")
print("=== CAUSAL ROBUSTNESS NOTES ===")
print(notes191[start192:start192 + 4700])

print("=== FUNCTION NAMES ===")
funcs192 = [
    n for n, o in list(globals().items())
    if inspect.isfunction(o) and not n.startswith("_")
]
print(sorted(funcs192))

print("=== KEY SAVED OBJECTS ===")
for prefix192 in ("c183", "c185", "r176", "r183", "r189", "r190"):
    rows192 = []
    for n192, o192 in sorted(list(globals().items())):
        if not n192.startswith(prefix192):
            continue
        if isinstance(o192, dict):
            desc192 = "keys=" + repr(list(o192.keys())[:20])
        elif hasattr(o192, "shape"):
            desc192 = str(tuple(o192.shape))
        elif inspect.isfunction(o192):
            desc192 = "function"
        else:
            desc192 = repr(o192)[:120]
        rows192.append(n192 + ": " + desc192)
    print(prefix192, "\n".join(rows192)[:1500])
```