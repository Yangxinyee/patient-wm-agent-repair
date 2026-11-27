The architecture confirms that dose changes reach readings through the recurrent state; there is no direct dose-to-reading head. The inventory was truncated, so I still lack the details needed for a safe edit. I’ll inspect the saved causal estimates and reconstruction functions next. Reliable held-out estimates and an accurate reconstruction would support using them; mismatches would rule that out.

```python
import numpy as np
import pandas as pd
import inspect
import types

print("=== FUNCTIONS ===")
fn_items = [(k, v) for k, v in list(globals().items())
            if isinstance(v, types.FunctionType) and not k.startswith("_")]
for k, v in fn_items:
    print(k, v.__code__.co_varnames[:v.__code__.co_argcount])

print("\n=== LATER ANALYSIS OBJECTS ===")
items = list(globals().items())
start = next((i for i, (k, _) in enumerate(items) if k == "code_id"), 0)
for k, v in items[start:]:
    if k.startswith("_") or isinstance(v, (types.ModuleType, types.FunctionType)):
        continue
    shape = getattr(v, "shape", None)
    if shape is not None and not callable(shape):
        print(k, type(v).__name__, tuple(shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), list(v)[:6])
    elif isinstance(v, list):
        print(k, "list", len(v))
    elif isinstance(v, (str, int, float, bool)) and len(str(v)) < 180:
        print(k, repr(v))

print("\n=== CAUSAL ESTIMATES ===")
for k in ("dyn_estimates", "dynamic_alternatives", "sensor_checks"):
    if k in globals():
        print(k, globals()[k])
print("panel columns:", panel.columns.tolist())
print(panel.head(3).to_string(index=False))

print("\n=== ORIGINAL FORWARD SUMMARIES ===")
for k in ("fA", "dA"):
    if k in globals():
        df = globals()[k]
        print(k, df.columns.tolist())
        print(df.head(5).to_string(index=False))
        print(df.describe(include="all").to_string())

print("\n=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("z_post", "z_prior", "z_in", "z_head", "embed.numeric",
                        "encode_action", "last_numeric", "attr_heads.value",
                        "core.prelude", "core.c_out", "core.out_norm")):
        print(name, SHAPES[name])

print("\n=== AVAILABLE FUNCTION SOURCE ===")
source_keys = ["causal_features", "mechanistic_shifts"]
source_keys += [k for k, _ in fn_items
                if any(t in k.lower() for t in ("local_step", "response", "posterior", "reconstruct"))]
for k in dict.fromkeys(source_keys):
    if k not in globals() or not callable(globals()[k]):
        continue
    try:
        src = inspect.getsource(globals()[k])
        print("\n", k, "\n", src[:6500])
    except (OSError, TypeError):
        fn = globals()[k]
        print(k, "source unavailable; referenced names:", fn.__code__.co_names)
print("Current edits:", edited())
```