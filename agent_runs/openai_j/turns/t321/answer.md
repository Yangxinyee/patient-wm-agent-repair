The encoder edit increased norepinephrine’s doubling response by only 0.021 mmHg, with little change elsewhere. That does not justify its large extra weight change. I’ll remove it and recover the causal estimates and saved value-head variants, then check factual safety. The remaining edit must stay within the 0.005-nat limit.

```python
import numpy as np
import re
from pathlib import Path

drop_edit("encode_action.mlp1.2.weight")
drop_edit("encode_action.mlp1.2.bias")
print("CURRENT EDITS:", edited())

notes279 = Path("/work/notes.md").read_text()
print("=== EARLY EVIDENCE ===")
print(notes279[:16000])
print("=== SAVED RECORD-BASED ESTIMATES ===")
for fn279 in ("kernel_targets.csv", "dml_results.csv", "dynamic_iv.csv", "sensor_iv.csv"):
    df279 = pd.read_csv("/work/" + fn279)
    print(fn279, "shape", df279.shape)
    print(df279.head(16).to_string(index=False))

print("=== SAVED WEIGHT VARIANTS ===")
for fn279 in ("validated_safe_backup268.npz", "calibrated_candidate268.npz",
              "pre_action_extrap_backup277.npz", "validated_combined_backup.npz"):
    with np.load("/work/" + fn279) as z279:
        print(fn279, [(k, z279[k].shape) for k in z279.files])
        print("max differences from current:",
              {k: float(np.max(np.abs(load(k) - z279[k])))
               for k in z279.files if k in SHAPES})

print("=== RELEVANT KERNEL NAMES ===")
print(sorted(k for k in globals()
             if re.search(r"(candidate|calib|kernel|strength|factual|nll|likelihood|restore)", k)
             and not k.endswith("279"))[:160])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 279: reject action extrapolation\n"
            "- Forward 35 / fwd_36: insulin -10.107/+5.453; NE +1.187/-1.752.\n"
            "- Compared with the four-tensor candidate, NE doubling improves only 0.021 mmHg.\n"
            "- Dropped both action-encoder edits: marginal rollout benefit does not justify their complexity.\n"
            "- Calibrated four-tensor value-head candidate remains installed; pooled factual validation pending.\n"
            "- Counts: 279 Python runs, 35 forward passes.\n")
```