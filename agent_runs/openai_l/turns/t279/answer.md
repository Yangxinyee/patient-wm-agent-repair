The tensor comparison confirms that the 0.3/0.3 head is still staged; the stronger grid candidates were never saved. I’ll now stage the 0.5/0.4 candidate, retaining the validated head for rollback. Its offline results predict a stronger insulin response at about +0.0039 factual NLL, which needs independent GPU confirmation.

```python
import numpy as np
import torch
import pandas as pd
from pathlib import Path

backup248 = {name: load(name).clone() for name in edited()}
T248 = np.asarray(grid242[(0.5, 0.4)]["T"], dtype=np.float32)

print("first-layer difference from validated architecture:",
      float((load("attr_heads.value_bin.0.weight") -
             torch.as_tensor(W1235, dtype=torch.float32)).abs().max()))
assert T248.shape == (193, 17)
assert np.isfinite(T248).all()
save_edit("attr_heads.value_bin.2.weight", torch.from_numpy(T248[:-1].T.copy()))
save_edit("attr_heads.value_bin.2.bias", torch.from_numpy(T248[-1].copy()))

check248 = np.vstack([
    load("attr_heads.value_bin.2.weight").numpy().T,
    load("attr_heads.value_bin.2.bias").numpy()[None, :]
])
print("saved candidate max difference:", float(np.max(np.abs(check248 - T248))))
ids248 = pd.read_csv("/work/fwd_18/factual_original.csv")["stay_id"].unique().tolist()
print("independent audit IDs:", ids248)
print("candidate offline scores:", grid242[(0.5, 0.4)]["score"])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 248 / stronger head actually staged\n"
            "Verified prior current head equals T235 (.3/.3); grid242 candidates "
            "were only exploratory. Saved grid242[(.5,.4)] final weights and bias; "
            "four-unit first layer and quiet upstream wire unchanged. backup248 "
            "preserves the validated .3/.3 candidate. Offline factual deltas "
            "+.002136/+.003901/+.003925; immediate I x2 slope -1.725, NE +1.961. "
            "Needs independent factual and paired rollout validation.\n")
```
