The smoother-gate screen barely changed the local dose response, but that does not establish its rollout behavior. The current edit remains the validated fallback. I’ll recover the exact helpers before making a reversible trial; only paired GPU results can confirm an improvement.

```python
from pathlib import Path
import inspect, re, numpy as np

print(Path("/model/ARCHITECTURE.md").read_text())

print("\nRELEVANT FUNCTIONS")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and re.search(
        r"gate|build|make|head|screen|cube|contrast|target|install|checkpoint", name, re.I
    ):
        print(name, inspect.signature(obj))
        print("  dependencies:", obj.__code__.co_names)

print("\nRELEVANT OBJECTS")
for name, obj in sorted(list(globals().items())):
    if re.search(r"checkpoint|gate|activity|r3|cube|^TP$|^CP$|^SP$", name, re.I):
        if isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:25])
        elif isinstance(obj, np.ndarray):
            print(name, "array", obj.shape, obj.dtype)
        elif hasattr(obj, "shape") and not inspect.isfunction(obj):
            print(name, "shape", tuple(obj.shape))
        elif isinstance(obj, (int, float, str, tuple, list)):
            print(name, repr(obj)[:240])

print("\nCURRENT FALLBACK EQUALITY")
for name in edited():
    a = np.asarray(load(name))
    b = np.asarray(validated_gated8_checkpoint[name])
    print(name, float(np.max(np.abs(a - b))))
print("No weights changed.")
```