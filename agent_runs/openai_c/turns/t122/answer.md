The latent-input gain was weak, but that path passes through token normalization. A targeted gain at `z_head` would instead reach reading predictions directly. I’ll test that distinct pathway using gain-4 state directions. Stronger correctly signed rollouts with acceptable factual NLL would support it; either failure would argue against it.

```python
import numpy as np
from pathlib import Path

# Fit a two-direction latent sensor using gain-4 states only.
s87 = np.load("/work/fwd_13/states_edited.npz")
z87 = s87["x"][:, 960:992].astype(np.float64)
dz87 = np.asarray(dose_gain4_72, dtype=np.float64)[:, 960:992]
mu87 = z87.mean(axis=0)
qmu87 = mu87 / max(np.linalg.norm(mu87), 1e-12)
p87 = np.eye(32) - np.outer(qmu87, qmu87)
c87 = z87.T @ z87 / len(z87)
ridge87 = 1e-3 * np.trace(c87) / 32
r87 = p87 @ np.linalg.solve(
    p87 @ c87 @ p87 + ridge87 * np.eye(32), p87 @ dz87.T
)
u87, sv87, _ = np.linalg.svd(r87, full_matrices=False)
latent_head_basis87 = u87[:, :2]

print("Latent dose-direction norms:", np.linalg.norm(dz87, axis=1))
print("Sensor singular values:", sv87)
print("Factual projected SD:", np.std(z87 @ latent_head_basis87, axis=0))
print("Dose projections:", dz87 @ latent_head_basis87)
print("z_head shape:", SHAPES["z_head.weight"])

clear_ops()
replay_ops(repair_candidates["numeric_gain4"])
for j in range(2):
    rank1("z_head.weight", latent_head_basis87[:, j], 9.0, side="in")
print("Installed:", ops())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 87: selective prior-head gain\n"
            "- Earlier latent-input gain was ineffective; testing a distinct direct reading pathway.\n"
            "- Numeric gain 4 plus z_head input gain 10 on a two-dimensional latent dose-sensor subspace.\n"
            "- Sensors use gain-4 states, regularized factual second moments, and a zero-mean constraint.\n"
            "- Posterior-state geometry may not transfer to prior draws; factual and rollout tests are decisive.\n"
            "- Forward passes used: 32; candidate is unvalidated.\n")
```
