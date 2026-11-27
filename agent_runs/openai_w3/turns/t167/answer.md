The architecture confirms that the reserved attention head can carry a signal into the recurrent state, but it currently carries none. The diagnostic output was truncated before the useful implementation details. I’ll recover those details in a smaller report; accurate dose contrasts and negligible recorded-dose payloads would justify testing this route.

```python
import inspect
import numpy as np
from pathlib import Path

print("=== TRANSPORT VARIABLES ===")
for name, obj in list(globals().items()):
    if any(s in name for s in ("145", "146", "147")):
        if isinstance(obj, np.ndarray):
            print(name, obj.shape, obj.dtype)
        elif hasattr(obj, "shape") and not inspect.isfunction(obj):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys", list(obj.keys())[:12])
        elif isinstance(obj, (int, float, bool)):
            print(name, obj)
        elif isinstance(obj, str):
            print(name, "string length", len(obj))
        elif isinstance(obj, (list, tuple)):
            print(name, repr(obj)[:180])

print("\n=== DECODER VARIABLES ===")
for name, obj in list(globals().items()):
    if any(s in name for s in ("104", "105")):
        if isinstance(obj, np.ndarray):
            print(name, obj.shape)
        elif isinstance(obj, dict):
            print(name, "keys", list(obj.keys())[:10])
        elif isinstance(obj, str):
            print(name, "string length", len(obj))
        elif isinstance(obj, (int, float, bool, list, tuple)):
            print(name, repr(obj)[:120])

print("\n=== HELPER IMPLEMENTATION REFERENCES ===")
for name in ("prelude_parts145", "fn146", "parts105",
             "next_view105", "action_vectors87", "state_update82"):
    obj = globals().get(name)
    if inspect.isfunction(obj):
        print(name, "names:", obj.__code__.co_names,
              "locals:", obj.__code__.co_varnames)
        print("constants:", repr(obj.__code__.co_consts)[:450])

print("\n=== SHORT IDENTIFICATION SUMMARY ===")
text150 = Path("/work/notes.md").read_text()
count150 = 0
for line in text150.splitlines():
    if any(s in line.lower() for s in
           ("instrument", "randomiz", "confound", "immediate effect",
            "impulse", "record-derived", "reference shifts")):
        print(line[:750])
        count150 += 1
        if count150 >= 10:
            break

print("\n=== POSTERIOR AND LATENT TENSOR SHAPES ===")
for name in NAMES:
    if "z_post" in name or "z_in" in name:
        print(name, SHAPES[name])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 150; forwards 10. Architecture recovered. "
        "Current five-tensor edit reserves an attention transport head "
        "with factual compensation but no causal payload. Recovering "
        "its compact implementation and retained nonlinear decoder "
        "before a payload contrast test. No weights changed.\n"
    )
```