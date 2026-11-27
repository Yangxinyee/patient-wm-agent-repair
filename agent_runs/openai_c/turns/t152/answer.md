The earlier head shear passed factual fit at strength 0.10 but barely changed rollouts. That test also altered the reading registers’ contribution, so it does not settle the query-only hypothesis. I’ll check the saved dose sensors and transform helpers next. A well-calibrated, nilpotent correction would support the compensated pilot; unstable sensors would argue against it.

```python
import numpy as np
import inspect
from pathlib import Path

print("=== HEAD-DESIGN VARIABLES (COMPACT) ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    if any(s in name.lower() for s in ("72", "73", "probe_ids", "head_actual")):
        obj = globals()[name]
        if hasattr(obj, "shape"):
            print(name, obj.shape)
        elif callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                print(name, "callable")
        elif isinstance(obj, (list, tuple, dict)):
            print(name, type(obj).__name__, "length", len(obj))
        else:
            print(name, repr(obj)[:100])

print("\n=== SENSOR AND SHEAR CALIBRATION ===")
for name in ("desired72", "W72", "D72", "T72", "R72", "S72"):
    obj = globals().get(name)
    if isinstance(obj, np.ndarray):
        if obj.size <= 16:
            print(name, obj)
        else:
            print(name, "norm", np.linalg.norm(obj),
                  "rank", np.linalg.matrix_rank(obj))
print("W72 @ D72.T =", W72 @ D72.T)
print("R72.T @ D72.T =", R72.T @ D72.T)
print("R72.T @ T72 =", R72.T @ T72)
print("S72 dose-response norms =", np.linalg.norm(S72 @ D72.T, axis=0))
print("S72 mean-state response norm =", np.linalg.norm(S72 @ O72.mean(axis=0)))

print("\n=== COMPENSATION TARGETS ===")
for name in NAMES:
    if (name.startswith("embed.") or name.startswith("last_") or
        name in ("encode.mlp1.0.weight", "encode_action.mlp1.0.weight") or
        (name.startswith("attr_heads.value_bin") and
         len(SHAPES[name]) == 2) or name.startswith("pending_mlp")):
        print(name, SHAPES[name])

print("\n=== TRANSFORM HELPERS ===")
for name in ("apply_lowrank_left_transform", "apply_lowrank_right_transform",
             "orthogonal_reflection_sequence", "orthogonal_rank1_factors",
             "effective_transform"):
    obj = globals().get(name)
    if callable(obj):
        print(name, inspect.signature(obj))
        try:
            print(inspect.getsource(obj)[:1700])
        except Exception:
            print("Locals:", obj.__code__.co_varnames)
            print("Names:", obj.__code__.co_names)
            print("Constants:", repr(obj.__code__.co_consts)[:500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 108: query-only calibration audit\n"
            "- Earlier uncompensated head shear had negligible rollout benefit.\n"
            "- Checking gain-4 sensors, shear orientation, and every input/register tensor.\n"
            "- No edits changed; numeric gain 4 remains installed. Forward passes: 40.\n")
```