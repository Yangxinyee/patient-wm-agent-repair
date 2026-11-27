The record-based probe targets are approximately −12.0/+6.0 for insulin and +4.7/−2.4 for norepinephrine, with uncertainty from causal identification and persistence fitting. The slot edit is far short of these. I’ll restore the factual-safe numeric-only candidate and inspect a more direct reading-head correction; it should strengthen dose responses without broadly disturbing recurrent state.

```python
import inspect
import numpy as np

print("=== REPLAY SIGNATURE ===", inspect.signature(replay_ops))
replay_ops(repair_candidates["numeric_gain4"])
print("Restored numeric gain 4:", ops())

print("=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in [
        "attr_heads.value", "query_norm", "last_numeric",
        "core.coda", "core.out_norm", "core.z_head", "type_emb"
    ]):
        print(name, SHAPES[name])

def compact_saved(x, depth=0):
    if isinstance(x, np.ndarray):
        return {
            "shape": x.shape,
            "norm": round(float(np.linalg.norm(x)), 5),
            "min": round(float(np.min(x)), 5),
            "max": round(float(np.max(x)), 5)
        }
    if isinstance(x, dict):
        if depth >= 2:
            return {"keys": list(x.keys())[:12]}
        return {str(k): compact_saved(v, depth + 1)
                for k, v in list(x.items())[:12]}
    if isinstance(x, (list, tuple)):
        return [compact_saved(v, depth + 1) for v in x[:5]]
    if isinstance(x, (float, int, str, np.number)):
        return x
    return type(x).__name__

for name in [
    "head_surrogates", "dose_state_coeff", "split_coeffs",
    "dose_directions", "state_correction_designs",
    "pilot_design", "pilot_basis", "state_file"
]:
    if name in globals():
        print(name, compact_saved(globals()[name]))

print("=== HELPER SIGNATURES ===")
for name in ["apply_lowrank_left_transform", "freeze_ops", "make_panel",
             "crossfit_residual_matrix", "record_based_rollouts"]:
    if name in globals() and callable(globals()[name]):
        print(name, inspect.signature(globals()[name]))
        try:
            print(inspect.getsource(globals()[name])[:1300])
        except (OSError, TypeError):
            pass

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python run 28\n"
        "- Recovered record-based sensor-masked probe targets: insulin -12.036/+6.018; "
        "NE +4.732/-2.366. These are estimates, not counterfactual truth.\n"
        "- Restored factual-safe numeric gain 4 alone; slot shear gave too little benefit.\n"
        "- Inspecting direct reading-head corrections. Forward passes: 15.\n"
    )
```