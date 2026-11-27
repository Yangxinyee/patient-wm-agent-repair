The saved directions come from dose innovations adjusted for prior states and record history. I’ll use their two-dimensional span to amplify recurrent drive, removing its component along the mean state. This is only a coordinate-based hypothesis: it is supported if rollouts strengthen substantially and refuted if they remain weak or factual fit deteriorates.

```python
import numpy as np
from pathlib import Path

# Keep the validated numeric edit; amplify only two recurrent output directions.
recurrent_ops_before103 = list(ops())
recurrent_raw103 = (
    dose_gain4_72[:, 192:960].reshape(2, 4, 192).mean(axis=1).T
)
recurrent_mean103 = np.asarray(h_mean_e).mean(axis=0)
recurrent_mean_unit103 = recurrent_mean103 / np.linalg.norm(recurrent_mean103)

recurrent_centered103 = recurrent_raw103 - np.outer(
    recurrent_mean_unit103,
    recurrent_mean_unit103 @ recurrent_raw103
)
recurrent_basis103, recurrent_r103 = np.linalg.qr(recurrent_centered103)
assert np.linalg.matrix_rank(recurrent_centered103) == 2
assert np.allclose(recurrent_basis103.T @ recurrent_basis103, np.eye(2))

recurrent_gain103 = 4.0
for direction in recurrent_basis103.T:
    rank1("core.b_in.weight", direction,
          recurrent_gain103 - 1.0, side="out")

h_centered103 = np.asarray(h_mean_e) - recurrent_mean103
projection_fraction103 = (
    np.sum((h_centered103 @ recurrent_basis103) ** 2)
    / np.sum(h_centered103 ** 2)
)
print("Raw direction norms:", np.linalg.norm(recurrent_raw103, axis=0))
print("After mean removal:", np.linalg.norm(recurrent_centered103, axis=0))
print("Fraction of saved mean-slot variance in span:", projection_fraction103)
print("Installed operations:", ops())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python run 103: selective recurrent-output pilot\n"
            "- Numeric gain 4 retained.\n"
            "- core.b_in output gain 4 on two orthonormal dose-innovation directions.\n"
            "- Directions use mean-slot gain-4 treatment coefficients, projected off the mean state.\n"
            "- Saved-state coordinates are a proxy for recurrent-drive coordinates; transfer is unproven.\n"
            "- Factual safety and rollout benefit are untested. Forward passes used: 39.\n")
```