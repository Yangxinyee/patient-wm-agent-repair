The decoders return dose-rate contrasts, and their latent directions are confined to the posterior mean. I’ll make a reversible, modest amplification of that pathway, leaving latent uncertainty and output heads unchanged. The next tests will show whether this increases the drug responses without exceeding the factual-fit tolerance.

```python
import numpy as np
from pathlib import Path

posterior_probe_strength = {"insulin": 4.0, "norepinephrine": 4.0}
posterior_probe_W0 = np.asarray(load_original("z_post.2.weight")).copy()
posterior_probe_b0 = np.asarray(load_original("z_post.2.bias")).copy()
posterior_probe_delta_W = {}
posterior_probe_delta_b = {}

W_new = posterior_probe_W0.astype(np.float64)
b_new = posterior_probe_b0.astype(np.float64)
for drug, strength in posterior_probe_strength.items():
    coef_d, intercept_d = post_full_models[
        (drug, "innovation", 10.0, 100.0)
    ]
    coef_d = np.asarray(coef_d, dtype=np.float64)
    direction_d = np.asarray(post_latent_directions[drug], dtype=np.float64)
    assert coef_d.shape == (192,) and direction_d.shape == (32,)
    dw = np.zeros_like(W_new)
    db = np.zeros_like(b_new)
    dw[:32] = np.outer(direction_d, coef_d)
    db[:32] = direction_d * float(intercept_d)
    posterior_probe_delta_W[drug] = dw
    posterior_probe_delta_b[drug] = db
    W_new += strength * dw
    b_new += strength * db

assert np.isfinite(W_new).all() and np.isfinite(b_new).all()
valid_H = np.asarray(post_H)[post_valid]
latent_delta = valid_H @ (W_new[:32] - posterior_probe_W0[:32]).T
latent_delta += b_new[:32] - posterior_probe_b0[:32]
print("Probe strengths:", posterior_probe_strength)
print("Factual posterior perturbation norm quantiles:",
      np.quantile(np.linalg.norm(latent_delta, axis=1), [0, .5, .9, .99, 1]))
print("Original posterior norm quantiles:",
      np.quantile(np.linalg.norm(np.asarray(post_z0)[post_valid, :32], axis=1),
                  [.5, .9, .99]))

save_edit("z_post.2.weight", W_new.astype(np.float32))
save_edit("z_post.2.bias", b_new.astype(np.float32))
print("ACTIVE EDITS:", edited())
print("TUNING FIRST 100 IDS:",
      sorted(tuning.stay_id.unique().tolist())[:100])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 64: reversible posterior-only probe\n"
            "- Added rank-two dose-contrast corrections to z_post.2 mean rows "
            "and corresponding biases; log-SD rows unchanged.\n"
            "- Strength 4 for both drugs, using innovation decoders with "
            "factual_weight=10 and alpha=100 and original latent dose directions.\n"
            "- This is a pathway probe, not an accepted repair. Require paired "
            "held-out factual fit and rollout response before retaining it.\n"
            "- Python runs: 64. Forward passes: 9.\n")
```
