Correcting the admission guard substantially improved the detector. The balanced candidate has lower factual activation and less insulin cross-response than the old detector, though halving remains somewhat stronger than doubling. I’ll install a modest norepinephrine correction and test it on the GPU. Factual fit and stable insulin readouts must both hold before increasing its strength.

```python
import numpy as np
import torch
from pathlib import Path

feature199 = fits198[(10.0, 1.0)]["feature"].astype(np.float64)
direction199 = archives196["ne_candidate82.npz"]["direction"].astype(np.float64)
beta199 = 0.8
ne_delta199 = beta199 * np.outer(direction199, feature199)
baseline_delta199 = archives196["baseline77.npz"]["delta"].astype(np.float64)
original_b199 = as_numpy194(load_original("core.b_in.weight")).astype(np.float64)
weight199 = original_b199 + baseline_delta199 + ne_delta199

print("SAVED DIRECTION CALIBRATION")
old199 = archives196["ne_trial103.npz"]
print("old beta", old199["beta"], "old strength", old199["strength"])
old_reconstructed199 = (
    float(old199["strength"]) *
    np.outer(old199["direction"], old199["feature"])
)
print("old matrix reconstruction error",
      float(np.max(np.abs(old_reconstructed199 - old199["delta"]))))
J199 = archives196["ne_candidate82.npz"]["jacobian"].astype(np.float64)
print("saved Jacobian direction responses",
      np.round(J199 @ direction199, 4))

print("\nNEW DETECTOR")
print("beta", beta199, "feature norm", float(np.linalg.norm(feature199)),
      "NE matrix norm", float(np.linalg.norm(ne_delta199)))
step199 = as_numpy194(step191).astype(np.float64).reshape(1, 1, -1)
perturb199 = (F197 @ ne_delta199.T) * step199
print("factual pre-coda slot RMS", float(np.sqrt(np.mean(perturb199 ** 2))))
print("initial-admission maximum",
      float(np.max(np.abs(perturb199[initial_admission198]))))
for key199, data199 in cf_train192.items():
    response199 = (data199["dE"] @ feature199).mean(axis=1)
    target199 = data199["dose_change"]
    print(key199, "dose slope",
          float(response199 @ target199 / (target199 @ target199)))

save_edit("core.b_in.weight", torch.from_numpy(weight199.astype(np.float32)))
np.savez(
    "/work/ne_candidate199.npz",
    ne_delta=ne_delta199.astype(np.float32),
    total_delta=(baseline_delta199 + ne_delta199).astype(np.float32),
    feature=feature199,
    direction=direction199,
    beta=np.array(beta199),
)
print("INSTALLED", edited())

with Path("/work/notes.md").open("a") as f199:
    f199.write(
        "\n## Run 199: install training-contrast NE trial\n"
        "Corrected initial-admission guard leaves 188 input directions. "
        "Selected factual weight10 / insulin guard1: held-out factual RMS=.3713, "
        "NE x2 slope=.9139, half slope=1.1599, insulin slopes=.0352/.1324. "
        "Installed baseline77 plus .8*direction82 outer normalized feature. "
        "Saved ne_candidate199.npz; unaccepted pending GPU factual and rollout "
        "checks. Python runs: 199; forwards: 22.\n"
    )
```