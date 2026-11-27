The architecture confirms that dose information enters through the action token and recurrent state. The earlier record estimates suggest glucose-lowering insulin effects and MAP-raising norepinephrine effects, unlike the original rollouts. The output was truncated before the replay helpers appeared, so I’ll retrieve those separately. Finding the convention error would let us calibrate an edit without relying on an inaccurate replay.

```python
import inspect, dis
print("=== REPLAY AND CALIBRATION HELPERS ===")
prefixes = ("r82", "r95", "r96", "r97", "r100", "r103", "r104",
            "r105", "r107", "r112", "r113")
for name in sorted(list(globals())):
    if name.startswith(prefixes):
        obj = globals()[name]
        if callable(obj):
            print(name, "FUNCTION", inspect.signature(obj),
                  "uses:", obj.__code__.co_names)
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "DICT", list(obj)[:8])
        elif isinstance(obj, str):
            print(name, "STRING", len(obj), obj[:60].replace("\n", " "))
print("\n=== CORE / READOUT TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("core.blocks", "core.coda", "core.c_out",
                        "core.out_norm", "core.prelude", "core.delta",
                        "core.a_in", "core.b_in", "z_head", "query_norm",
                        "last_numeric", "attr_heads.value_bin")):
        print(name, SHAPES[name])
print("\n=== LIVE SOURCE STRINGS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if isinstance(obj, str) and (
        "def r112" in obj or "def r113" in obj or
        ("def " in obj and "core.blocks.0.0" in obj)
    ):
        print(name, obj[:7000])
```
