The correction still strengthens insulin responses on the last 100 stays, though halving remains weak. I’ll compare that result with the dose-history-derived target and inspect whether smoothing the activity gate could help. This hypothesis would be supported by a better halving response without losing the doubling response or factual fit.

```python
import inspect
import numpy as np
from pathlib import Path

print("TARGET HELPERS:")
for name273 in ["causal_shift_table", "matched_causal_targets163",
                "sc_evaluate", "sc_cache_inputs", "screen_value_candidate",
                "graft_probs", "make_contrast_graft"]:
    f273 = globals()[name273]
    print(name273, inspect.signature(f273))
    print("  names:", f273.__code__.co_names)
    print("  string constants:",
          [c for c in f273.__code__.co_consts if isinstance(c, str)][:12])

print("\nACTIVITY CONFIG:")
for k273, v273 in activity_gate_config.items():
    if isinstance(v273, np.ndarray):
        print(k273, v273.shape, "norm", float(np.linalg.norm(v273)))
    else:
        print(k273, v273)
print("activity_graft_rows:", activity_graft_rows)

print("\nFACTUAL CACHE SHAPES:")
for k273, v273 in sc_fact["first"].items():
    print(k273, np.shape(v273))
print("\nPANEL OBJECTS:")
for k273 in ["P", "Pt", "P_tuning", "tuning_panel", "panel"]:
    if k273 in globals():
        v273 = globals()[k273]
        print(k273, type(v273).__name__, getattr(v273, "shape", None),
              list(v273.columns)[:12] if hasattr(v273, "columns") else "")

with Path("/work/notes.md").open("a") as f273:
    f273.write(
        "\n## Independent rollout check; Python 273, forwards 33/50\n"
        "- Current weights remain exactly validated_gated8_checkpoint.\n"
        "- Last100 tuning rollout: insulin x2 -7.540, half +2.237; "
        "original -1.571, +0.649. NE x2 +0.027, half -0.433.\n"
        "- Replacement blends reduce cached NLL but weaken local contrasts; "
        "none has been installed.\n"
        "- Investigating a smoother activity gate; any trial remains reversible "
        "and requires exact GPU validation.\n")
```
