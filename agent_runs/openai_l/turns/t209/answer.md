The edit only modestly improves insulin responses and barely changes norepinephrine responses. Scaling this readout further is unlikely to solve the problem within the factual limit. I’ll inspect the captured head inputs for a more selective correction—one that responds strongly to dose changes but stays quiet on factual records.

```python
import inspect, torch
from pathlib import Path

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward 12 / weak causal benefit\n"
        "Paired rollout on the 100-stay audit: original shifts "
        "[-3.680, +2.018, +0.465, -0.871]; current shifts "
        "[-4.302, +2.434, +0.496, -0.912] for I x2/half, NE x2/half. "
        "Current factual delta +0.00263. Final-layer-only repair is "
        "insufficient, especially for NE. Investigating selective "
        "head-feature construction rather than increasing its scale.\n"
    )

for name in ["decoder_base180", "decoder_capture180", "f176", "f177"]:
    obj = globals().get(name)
    if obj is not None:
        print("\nFUNCTION", name)
        try:
            print(inspect.getsource(obj)[:2400])
        except (OSError, TypeError):
            print("names:", obj.__code__.co_names)
            print("constants:", str(obj.__code__.co_consts)[:700])

for name in ["base180", "case180", "readout179", "targets179"]:
    obj = globals().get(name)
    print("\nOBJECT", name)
    if isinstance(obj, dict):
        items = list(obj.items())
        if name in ["readout179", "targets179"]:
            items = items[:2]
        for key, val in items:
            print(key, type(val).__name__,
                  tuple(val.shape) if hasattr(val, "shape") else str(val)[:140])
    elif obj is not None:
        print(type(obj).__name__)

print("\nFEATURE / CACHE OBJECT NAMES:")
for name, obj in sorted(list(globals().items())):
    if re.match(r"^(cf_|cache|weights|states|train|tune|dec_)", name):
        if isinstance(obj, dict):
            print(name, "dict", list(obj)[:6])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif callable(obj):
            try:
                print(name, inspect.signature(obj))
            except (ValueError, TypeError):
                pass

print("\nHEAD TENSORS:")
for name in NAMES:
    if name.startswith("attr_heads.value_bin"):
        print(name, SHAPES[name])

print("\nROW SAMPLE:")
for name in ["rows", "rows173", "rows179", "meta173", "meta179"]:
    obj = globals().get(name)
    if obj is not None:
        print(name, type(obj).__name__, str(obj[:3])[:500])
```