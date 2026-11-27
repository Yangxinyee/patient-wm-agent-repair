The architecture confirms that dose information passes through several nonlinear stages before reaching the value head. The helper listing was truncated, so it did not identify the usable replay functions. I’ll recover their names and recent diagnostic objects compactly, then test a small downstream correction. A factual-safe increase in dose sensitivity would support that approach.

```python
import inspect, re
from pathlib import Path

lines240 = ["=== FUNCTIONS: PARAMETER NAMES ONLY ==="]
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            args = ", ".join(inspect.signature(obj).parameters.keys())
        except Exception:
            args = "?"
        lines240.append(f"{name}({args})")

lines240.append("\n=== RECENT REPLAY OBJECTS ===")
recent240 = re.compile(r"(?:22[8-9]|23[0-8])(?:_|$)")
for name, obj in sorted(list(globals().items())):
    if recent240.search(name) and not inspect.isfunction(obj):
        if isinstance(obj, dict):
            lines240.append(f"{name}: dict keys={list(obj)[:18]}")
        elif isinstance(obj, str):
            lines240.append(f"{name}: str[{len(obj)}] {obj[:70]!r}")
        elif hasattr(obj, "shape"):
            lines240.append(f"{name}: {type(obj).__name__}{tuple(obj.shape)}")
        elif isinstance(obj, (list, tuple)):
            item_types = [type(x).__name__ for x in obj[:5]]
            lines240.append(f"{name}: {type(obj).__name__}[{len(obj)}] {item_types}")
        else:
            lines240.append(f"{name}: {type(obj).__name__}")

lines240.append("\n=== OTHER REPLAY / HEAD / FEATURE DICTIONARIES ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, dict) and any(s in name.lower() for s in
        ["replay", "capture", "head", "candidate158", "backup218", "feature"]):
        lines240.append(f"{name}: keys={list(obj)[:20]}")

print("\n".join(lines240)[:9000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 240: compact interface recovery\n"
            "- Architecture confirms prelude-to-recurrent-to-coda attenuation of action signals.\n"
            "- Previous inventory was truncated by tensor-valued function defaults.\n"
            "- Listing parameter names and recent replay objects without defaults.\n"
            "- No weights changed. Python runs: 240; forward passes: 17.\n")
```