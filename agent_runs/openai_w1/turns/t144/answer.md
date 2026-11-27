The coda inversion is numerically stable, but a simple coefficient correction does not explain the remaining error. I won’t treat this replay as validated. I’ll recover the record-based effect estimates and factual-head helpers, then use exported states for factual constraints and forward passes to test any proposed edit.

```python
from pathlib import Path
import inspect, re

_text132 = Path("/work/notes.md").read_text()
print("=== EARLY RECORD-BASED FINDINGS ===")
print(_text132[:6200])

print("\n=== HEAD, EFFECT, AND CALIBRATION OBJECTS ===")
_g132 = dict(globals())
_matches132 = []
for name, obj in sorted(_g132.items()):
    if name.startswith("_"):
        continue
    if (name.startswith(("ce_", "repair_")) or
        any(s in name.lower() for s in
            ("kernel", "impulse", "causal", "effect", "head_x", "head_h",
             "head_input", "target_shift", "coef"))):
        if inspect.isfunction(obj):
            desc = str(inspect.signature(obj))
        elif hasattr(obj, "shape"):
            desc = type(obj).__name__ + " " + str(tuple(obj.shape))
        elif isinstance(obj, dict):
            desc = "dict " + str(len(obj)) + " " + repr(list(obj)[:5])
        elif isinstance(obj, (list, tuple)):
            desc = type(obj).__name__ + " length " + str(len(obj))
        elif isinstance(obj, (str, int, float, bool)):
            desc = repr(obj)[:130]
        else:
            continue
        _matches132.append(name + ": " + desc)
print("\n".join(_matches132)[:2700])

print("\n=== CURRENT EDITS ===")
print(edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n\n## Run 132\n"
        "Coda inversion is stable, but no simple recurrent coefficient repair "
        "explains replay discrepancy. Will use exported states for exact factual "
        "constraints, and regard replay-based sensitivities as approximate until "
        "GPU validation. Recovering record evidence and factual head helpers. "
        "No weights edited. Counts: 132 Python runs, 6 forward passes.\n"
    )
```