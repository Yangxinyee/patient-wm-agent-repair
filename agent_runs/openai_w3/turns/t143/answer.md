The recovered notes show substantially larger record-derived effects than the original model predicts: insulin lowers glucose, while norepinephrine raises MAP. Those are estimates from the records, not medical assumptions. The output was truncated before the helper inventory, so I’ll retrieve that compactly to choose an effective, reversible injection point.

```python
import re, inspect
from pathlib import Path

print("=== HELPER SIGNATURES ===")
for nm130, ob130 in list(globals().items()):
    if inspect.isfunction(ob130) and getattr(ob130, "__module__", "") == "__main__":
        try:
            sig130 = str(inspect.signature(ob130))
        except Exception:
            sig130 = "?"
        print(nm130, sig130[:180])

print("\n=== INJECTION TENSOR SHAPES ===")
for nm130 in NAMES:
    if nm130.startswith(("core.prelude.", "core.z_post", "core.z_in", "core.z_prior")):
        print(nm130, SHAPES[nm130])

print("\n=== CACHE TYPES ===")
for nm130 in ("prelude_cache88", "contrast_fits88", "pool_fits89",
              "policy_fits103", "decoder103", "action_coef105",
              "value_directions105", "target105", "pick105", "abs105"):
    ob130 = globals().get(nm130)
    if isinstance(ob130, dict):
        print(nm130, "dict", list(ob130)[:16])
    elif hasattr(ob130, "shape"):
        print(nm130, type(ob130).__name__, tuple(ob130.shape))
    else:
        print(nm130, type(ob130).__name__)

with Path("/work/notes.md").open("a") as f130:
    f130.write("\nPython run 130; forwards 7. Restart recovered architecture and early identification evidence. No edits. Original rollout is severely attenuated relative to record-derived insulin and NE kernels; proceeding to a reversible action-path pilot with authoritative factual validation, not further exact-readout reconstruction.\n")
```
