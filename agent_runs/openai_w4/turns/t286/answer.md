The retained candidates let me change only the two posterior tensors controlling insulin’s innovation channel. I’ll try gain 1.25 with gate shift 28, between the responsive gate-26 version and the nearly silent gate-40 version. Its CPU NLL increase was 0.00345; GPU tests must confirm both factual safety and better dose calibration.

```python
import numpy as np
import torch
from pathlib import Path

r249_key = (1.25, 28.0)
r249_candidate = r244_candidates[r249_key]
r249_names = ("z_post.0.weight", "z_post.0.bias")
r249_backup = {n: torch.as_tensor(load(n)).clone() for n in edited()}

# Ensure the candidate does not silently alter any other retained repair.
for n in edited():
    if n not in r249_names:
        assert torch.equal(torch.as_tensor(load(n)),
                           torch.as_tensor(r249_candidate[n])), n

for n in r249_names:
    t = torch.as_tensor(r249_candidate[n], dtype=torch.float32)
    assert tuple(t.shape) == tuple(SHAPES[n])
    assert torch.isfinite(t).all()
    save_edit(n, t.clone())

print("Saved insulin gain / total gate shift:", r249_key)
print("CPU screening result:", r244_results[r249_key])
print("Only changed relative to previous submission:", list(r249_names))
print("Validation stay count:", len(r246_ids))

print("RETAINED LONGITUDINAL KERNELS:")
for drug, obj in kernels137.items():
    if isinstance(obj, dict):
        print(drug, "keys", list(obj.keys()))
        for k, v in obj.items():
            if hasattr(v, "shape") and np.asarray(v).size <= 36:
                print(" ", k, np.asarray(v).round(6).tolist())
            elif isinstance(v, (int, float, str)):
                print(" ", k, v)
    elif hasattr(obj, "shape"):
        print(drug, np.asarray(obj).round(6).tolist())
    else:
        print(drug, repr(obj)[:250])

print("RUN-238 MATCHED TARGETS:", repr(matched_targets238)[:1400])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 249\n"
            "Saved retained insulin innovation-only gain 1.25 / total gate shift 28 "
            "candidate from Run 244. Only z_post.0.weight/bias changed; checked all other "
            "edited tensors equal the candidate. Backup r249_backup preserves gate-40 "
            "submission. CPU screen delta +.00344586. GPU factual and rollout "
            "validation pending. This is a computational calibration experiment, "
            "not new evidence about ICU effects.\n")
```
