The architecture exposes an action-to-posterior pathway that may allow a more selective repair than changing the value head globally. The kernel fit alone still does not establish the sustained effect. I’ll inspect the retained dose-innovation and action-gate tests; this approach is promising only if it distinguishes unexpected doses on held-out stays without activating on ordinary measurement bundles.

```python
import numpy as np
import pandas as pd

def summary91(x, depth=0):
    if isinstance(x, dict):
        return {str(k): summary91(v, depth + 1) for k, v in list(x.items())[:12]} if depth < 2 else list(x)[:12]
    if isinstance(x, pd.DataFrame):
        return {"type": "DataFrame", "shape": x.shape, "columns": list(x.columns)[:22]}
    if isinstance(x, (np.ndarray,)):
        return {"shape": x.shape, "dtype": str(x.dtype)}
    if hasattr(x, "shape"):
        return {"type": type(x).__name__, "shape": tuple(x.shape)}
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "len": len(x),
                "first": summary91(x[0], depth + 1) if len(x) else None}
    if callable(x):
        return {"function": str(inspect.signature(x))}
    return str(x)[:160]

for name in [
    "gate87", "gate88", "gate_models88", "gate_rows88", "kernel_models88",
    "_moment_inputs", "persistence_repair", "provisional_kernels",
    "head_innovation_probes", "action_fixture_repair", "repair_bank",
    "post_input", "state_panel", "state_panel_idx", "panel_stays_repair"
]:
    if name in globals():
        print(name, summary91(globals()[name]))

print("\nRECENT DIAGNOSTIC OBJECTS")
for name, obj in list(globals().items()):
    if name.endswith(("87", "88")) and not name.startswith("_"):
        print(name, summary91(obj, 2))

print("\nLARGE RECORD TABLES")
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and len(obj) >= 10000:
        print(name, obj.shape, list(obj.columns)[:14])

print("\nPOSTERIOR / CORE WEIGHTS")
for name in NAMES:
    if name.startswith(("core.z", "core.flow", "core.a_in", "core.delta", "core.b_in",
                        "core.c_out", "core.out", "core.prelude_norm")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 91\n"
            "Architecture confirms an explicit action/posterior pathway. "
            "Inspecting retained innovation and action-gate validation to assess "
            "a localized posterior repair; sustained causal targets remain provisional. No edits.\n")
```
