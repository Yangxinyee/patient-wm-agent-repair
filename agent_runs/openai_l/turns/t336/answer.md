The recovered records support immediate effects near −1.95 for insulin and +3.1 for norepinephrine, independently of the model’s predictions. The saved (.45, .4, 8) solution moves all four local dose responses closer to those anchors. I’ll stage it for testing; it must improve rollouts and preserve the overall factual-loss limit to replace the current candidate.

```python
import torch
import numpy as np

key291 = (0.45, 0.4, 8.0)
T291 = np.asarray(grid286[key291]["T"], dtype=np.float64)
assert T291.shape == current_T290.shape == (193, 17)
assert np.isfinite(T291).all()

save_edit("attr_heads.value_bin.2.weight",
          torch.from_numpy(T291[:-1].T.astype(np.float32).copy()))
save_edit("attr_heads.value_bin.2.bias",
          torch.from_numpy(T291[-1].astype(np.float32).copy()))

print("STAGED:", key291)
print("OFFLINE_SCORES:", grid286[key291]["score"])
print("HEAD_CHANGE_NORM:", float(np.linalg.norm(T291 - current_T290)))
print("RESTORE_BACKUP:", "backup290")
print("EDITED_TENSORS:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 291 / stronger joint-dose head staged\n"
            "Staged grid286(.45,.4,8), changing only final value-head weights "
            "and bias. Local dose slopes improve in all four contrasts relative "
            "to grid286(.35,.3,8); offline factual deltas "
            "+.001528/+.004597/+.004303. backup290 preserves the audited "
            "df0778bbce candidate. Pending paired GPU factual and rollout tests; "
            "local slope agreement is not proof of rollout agreement.\n")
```
