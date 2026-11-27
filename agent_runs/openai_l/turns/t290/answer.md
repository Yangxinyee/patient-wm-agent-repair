The notebook confirms that insulin’s negative effect was supported by dose-threshold comparisons, while conflicting state-space estimates were rejected. The output was truncated before the fitting details. I’ll retrieve those details compactly; a more selective dose feature is useful only if it strengthens the intervention response without worsening independent factual predictions.

```python
import inspect, re

print("=== FITTING FUNCTIONS ===")
for n257, o257 in sorted(list(globals().items())):
    if callable(o257) and hasattr(o257, "__code__"):
        c257 = o257.__code__
        if any(s in n257 for s in ("214", "219", "231", "234", "235", "242", "254")) or any(
            s in c257.co_names for s in ("paired214", "directions207", "grid242")
        ):
            print(n257, inspect.signature(o257), "globals:", ",".join(c257.co_names)[:500])

print("\n=== FITTING OBJECTS ===")
wanted257 = [
    "Xraw204", "X200", "Xraw228", "H228", "L228", "fitmask", "valmask",
    "paired214", "units235", "rows235", "T235", "grid242", "backup248",
    "kernels19", "targets179", "dosecenter214", "pruneorder214"
]
for n257 in wanted257:
    if n257 in globals():
        o257 = globals()[n257]
        if isinstance(o257, np.ndarray):
            print(n257, o257.shape, o257.dtype)
        elif isinstance(o257, dict):
            print(n257, "keys", list(o257)[:18])
        else:
            print(n257, type(o257).__name__, str(o257)[:150])

print("\n=== ACCEPTED RECORD EVIDENCE ===")
sections257 = re.split(r"\n(?=## )", notes256)
selected257 = [
    s for s in sections257
    if any(w in s.lower() for w in
           ("kernel19", "distributed", "accepted record", "persistence", "run 19", "run 18", "run 20", "run 21"))
]
for s257 in selected257[:10]:
    print(s257[:650])

print("\n=== RECENT FITTING NOTES ===")
for s257 in sections257:
    if any(w in s257.split("\n", 1)[0] for w in ("231", "232", "233", "234", "242", "243", "244", "245", "246", "250", "251")):
        print(s257[:550])
```
