The pilot worsened factual NLL by 0.00897 nats, so its current strength fails the constraint. I’ll remove the head correction and recover its saved transform for a weaker test. A weaker version is worth keeping only if it passes NLL and improves the dose rollouts.

```python
import inspect
from pathlib import Path
import numpy as np

failed_pilot28_ops = [
    dict(o, vector_array=np.load(Path("/work") / o["vector"]).copy())
    for o in ops()
]
failed_pilot28_head_transform = np.eye(192)
for _o75 in failed_pilot28_ops:
    if _o75["name"] == "attr_heads.value_bin.0.weight":
        _v75 = _o75["vector_array"]
        _v75 = _v75 / np.linalg.norm(_v75)
        failed_pilot28_head_transform = (
            failed_pilot28_head_transform
            @ (np.eye(192) + _o75["alpha"] * np.outer(_v75, _v75))
        )

clear_ops()
rank1("embed.numeric.weight",
      failed_pilot28_ops[0]["vector_array"], 3.0, side="in")
print("Restored numeric gain 4:", ops())

print("\n=== TRANSFORM HELPERS ===")
for _k75 in sorted(list(globals())):
    if callable(globals()[_k75]) and any(
        s in _k75.lower() for s in ("apply_lowrank", "restore", "install", "save_ops")
    ):
        try:
            print(_k75, inspect.signature(globals()[_k75]))
        except (TypeError, ValueError):
            pass

print("\n=== PILOT MATRICES ===")
print("head_basis73 shape:", head_basis73.shape)
print("head_reduced73:\n", head_reduced73)
print("Basis-space reconstruction error:",
      np.linalg.norm(failed_pilot28_head_transform -
                     (np.eye(192) + head_basis73
                      @ (head_reduced73 - np.eye(4)) @ head_basis73.T)))

print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:6500])

with Path("/work/notes.md").open("a") as _f75:
    _f75.write(
        "\n## Forward 28 / Python run 75: recalibrated pilot too strong\n"
        "- Strength 0.20: factual NLL 1.64358 vs 1.63461, increase +0.00897.\n"
        "- Saved its operation vectors and effective head transform in kernel variables.\n"
        "- Removed the head correction; numeric gain 4 alone is restored.\n"
        "- Inspecting the transform for a weaker, factual-safe pilot. Forwards used: 28.\n"
    )
```