Independent scaling creates a safer trade-off: less insulin overshoot and a stronger MAP response without increasing local factual loss beyond the limit. I’ll test insulin strength 0.90 and MAP gain 9.5 on the GPU, keeping the validated edit as a backup. The candidate must pass factual fit and improve the paired rollouts to be retained.

```python
import numpy as np
import torch

backup243 = {n: torch.as_tensor(load(n), dtype=torch.float32).clone()
             for n in head177_names}
np.savez("/work/validated_gain8_backup243.npz",
         **{n: t.numpy() for n, t in backup243.items()})

candidate243 = make242(0.90, 9.5)
tr243 = head177_eval(trainmc242, candidate243)
tu243 = head177_eval(tunemc242, candidate243)
dt243 = tr243["nll"] - base_train242
du243 = tu243["nll"] - base_tune242
summary243 = {
    "insulin_strength": 0.90,
    "MAP_gain": 9.5,
    "fit_dNLL": float(dt243[fit236].mean()),
    "heldout_dNLL": float(dt243[~fit236].mean()),
    "tuning_dNLL": float(du243.mean())
}
print("CANDIDATE", summary243)

assert summary243["heldout_dNLL"] < 0.0045
assert summary243["tuning_dNLL"] < 0.0045
for n, t in candidate243.items():
    assert tuple(t.shape) == tuple(SHAPES[n])
    assert torch.isfinite(t).all()
for n, t in candidate243.items():
    save_edit(n, t.numpy())

np.savez("/work/independent_candidate243.npz",
         **{n: t.numpy() for n, t in candidate243.items()})
print("Installed tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 243: independent-strength candidate installed\n"
        "- Insulin output strength 0.90; calibrated MAP gain 9.5.\n"
        "- Local factual checks: " + repr(summary243) + ".\n"
        "- Previous GPU-validated gain-8 edit backed up to validated_gain8_backup243.npz.\n"
        "- Pending independent GPU factual and paired rollout checks; only four value-head tensors edited.\n"
    )
```