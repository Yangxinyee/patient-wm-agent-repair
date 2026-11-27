The saved candidates are ready to test. I’ll install the calibrated version with insulin strength 0.90 and MAP gain 8.5, retaining the validated edit as a fallback. Its cached loss is promising; GPU factual checks and paired rollouts must confirm that the extra MAP response is worth the change.

```python
import torch
import numpy as np
import pandas as pd
from pathlib import Path

key268 = "i0.9_g8.5_soft_low"
candidate268 = r265_candidates[key268]
assert set(candidate268) == set(edited())
for name, tensor in candidate268.items():
    arr = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    assert tuple(arr.shape) == tuple(SHAPES[name])
    assert np.isfinite(arr).all()

backup268 = {}
for name in edited():
    tensor = load(name)
    backup268[name] = (tensor.detach().cpu().numpy().copy()
                      if torch.is_tensor(tensor) else np.asarray(tensor).copy())
np.savez("/work/validated_safe_backup268.npz", **backup268)

for name, tensor in candidate268.items():
    save_edit(name, tensor.clone() if torch.is_tensor(tensor) else tensor.copy())
np.savez("/work/calibrated_candidate268.npz", **{
    name: tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    for name, tensor in candidate268.items()
})

print("INSTALLED:", key268)
print(r265_frontier.loc[r265_frontier["key"] == key268].to_string(index=False))
print("\nHIGHER-GAIN SCREEN:")
print(r265_frontier.loc[r265_frontier["MAP_gain"] >= 10,
      ["key", "cv_dNLL", "tune_dNLL"]].to_string(index=False))

ids268 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].drop_duplicates()
print("\nTUNING FIRST 100 IDS:", ids268.iloc[:100].astype(int).tolist())
print("Tuning stays:", len(ids268))

text268 = Path("/work/notes.md").read_text()
sections268 = re.split(r"(?=^## )", text268, flags=re.M)
evidence268 = [s for s in sections268
               if re.search(r"instrument|identif|randomiz|causal evidence", s, re.I)]
print("\nEARLY IDENTIFICATION NOTES:")
for section in evidence268[:5]:
    print(section[:650])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 268: calibrated gain-8.5 candidate installed\n"
            "- Installed i0.9_g8.5_soft_low from the closed-form statistical screen.\n"
            "- Insulin strength remains 0.90; MAP gain rises from 7 to 8.5, with frozen-feature calibration.\n"
            "- Previous GPU-validated edit saved as validated_safe_backup268.npz.\n"
            "- Only four value-head tensors changed; GPU factual and paired rollout validation pending.\n")
print("\nCurrent edits:", edited())
```