At matched hours, the current candidate still falls short of the record-derived targets in all four shifts. I’ll trial 20% more insulin-component strength and 35% more norepinephrine-component strength, keeping an exact fallback. This is a calibration hypothesis: it is acceptable only if direct factual tests retain the budget and rollouts move closer to the targets.

```python
import numpy as np
import torch
import hashlib

fallback_B219 = current_B215.numpy().copy()
trial_scale_ins219 = 1.20
trial_scale_ne219 = 1.35

candidate_B219 = (
    original_B215.numpy().astype(np.float64)
    + trial_scale_ins219 * delta_ins218
    + trial_scale_ne219 * delta_ne218
).astype(np.float32)

assert candidate_B219.shape == tuple(original_B215.shape)
assert np.isfinite(candidate_B219).all()
assert np.max(np.abs(
    original_B215.numpy().astype(np.float64)
    + delta_ins218 + delta_ne218 - fallback_B219
)) < 2e-5

save_edit("core.b_in.weight", torch.from_numpy(candidate_B219))
assert edited() == ["core.b_in.weight"]
assert np.array_equal(
    torch.as_tensor(load("core.b_in.weight")).numpy(), candidate_B219
)

print("INSTALLED CONTROLLED GAIN TRIAL")
print("component multipliers", trial_scale_ins219, trial_scale_ne219)
print("hash", hashlib.sha256(candidate_B219.tobytes()).hexdigest()[:12])
print("edit norm", np.linalg.norm(candidate_B219-original_B215.numpy()))
print("fallback exact", np.array_equal(fallback_B219, current_B215.numpy()))
print("NEXT: paired factual tests before accepting or interpreting this trial.")

with open("/work/notes.md", "a") as f219:
    f219.write(
        "\n## Run219: controlled gain trial\n"
        "Installed candidate_B219 = original + 1.20*delta_ins218 + "
        "1.35*delta_ne218, only core.b_in.weight. Exact fallback_B219 "
        "is candidate_B212 (hash 973acf3505). Motivation: on untouched "
        "matched reached hours current shifts are -11.26594/+4.61005 "
        "versus record targets -14.87714/+7.43857; NE +5.55432/-2.54535 "
        "versus +8.69952/-4.34976. Targets assume the fitted latent persistence "
        "and identified drug slopes. This stronger trial is NOT accepted; "
        "require paired factual budget and rollout validation.\n"
    )
```
