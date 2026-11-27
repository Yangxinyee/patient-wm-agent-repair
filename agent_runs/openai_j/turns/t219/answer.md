The saved record-based targets are approximately −10.5/+5.3 for insulin and +3.8/−1.9 for norepinephrine on the first 100 tuning stays. These remain statistical estimates, not known truth. I’ll install the safer calibrated MAP candidate while preserving the insulin correction. A factual increase below +0.005 and norepinephrine rollout shifts closer to these targets would support keeping it.

```python
import numpy as np, pandas as pd, torch
from pathlib import Path

def np197(t):
    return t.detach().cpu().numpy().copy() if torch.is_tensor(t) else np.asarray(t).copy()

backup197 = {name: np197(load(name)) for name in head177_names}
np.savez("/work/validated_combined_backup.npz", **backup197)

candidate197 = head177_make(map192_candidates[(10.0, 6.0)])
for name, tensor in candidate197.items():
    array = np197(tensor)
    assert array.shape == tuple(SHAPES[name]), (name, array.shape, SHAPES[name])
    assert np.isfinite(array).all(), name

for neuron in [170, 89]:
    print("Insulin output-column change", neuron,
          np.max(np.abs(np197(candidate197[head177_names[2]])[:, neuron]
                        - backup197[head177_names[2]][:, neuron])))

for name, tensor in candidate197.items():
    save_edit(name, np197(tensor))
np.savez("/work/map_cal_candidate_v2.npz",
         **{name: np197(tensor) for name, tensor in candidate197.items()})

fact197 = pd.read_csv("/work/fwd_18/factual_original.csv")
ids197 = sorted(fact197["stay_id"].unique().astype(int).tolist())
print("Validation stay count:", len(ids197))
print("STAYS:", ids197)
print("Edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 197: calibrated MAP candidate installed\n"
            "- Installed head177_make(map192_candidates[(10,6)]), preserving insulin neurons.\n"
            "- Backed up forward-19 validated edit to validated_combined_backup.npz.\n"
            "- New candidate saved to map_cal_candidate_v2.npz; only four value-head tensors edited.\n"
            "- Local held-out training/tuning dNLL: +0.00187/+0.00317.\n"
            "- Pending GPU factual and dose-rollout validation; 19 forward passes used.\n")
```
