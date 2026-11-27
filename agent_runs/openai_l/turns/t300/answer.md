The new dose decoders have less factual variation than the earlier insulin decoder, and the conservative head predicts stronger norepinephrine contrasts. I’ll stage that candidate only after verifying the current weights against the saved baseline. A paired GPU factual test will determine whether its offline loss estimate holds.

```python
import torch
import pandas as pd
from pathlib import Path
import numpy as np

def array267(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

units267 = np.asarray(arch263["units"], dtype=int)
oldrows267 = np.asarray(st260["arch"]["rows"])
newrows267 = np.asarray(arch263["rows"])
oldT267 = np.asarray(grid242[(0.5, 0.4)]["T"])
newT267 = np.asarray(trial263[(0.3, 0.3)]["T"])

W0267 = array267(load("attr_heads.value_bin.0.weight")).copy()
b0267 = array267(load("attr_heads.value_bin.0.bias")).copy()
W2267 = array267(load("attr_heads.value_bin.2.weight"))
b2267 = array267(load("attr_heads.value_bin.2.bias"))
checks267 = {
    "first_weight": float(np.max(np.abs(W0267[units267] - oldrows267[:, :-1]))),
    "first_bias": float(np.max(np.abs(b0267[units267] - oldrows267[:, -1]))),
    "final_weight": float(np.max(np.abs(W2267 - oldT267[:-1].T))),
    "final_bias": float(np.max(np.abs(b2267 - oldT267[-1]))),
}
print("Baseline mapping checks:", checks267)
assert max(checks267.values()) < 2e-5, "Mapping mismatch: no candidate saved."
assert newrows267.shape == (4, 193) and newT267.shape == (193, 17)
assert np.isfinite(newrows267).all() and np.isfinite(newT267).all()

backup267 = {n: load(n).clone() for n in edited()}
W0267[units267] = newrows267[:, :-1]
b0267[units267] = newrows267[:, -1]
for n267, a267 in [
    ("attr_heads.value_bin.0.weight", W0267),
    ("attr_heads.value_bin.0.bias", b0267),
    ("attr_heads.value_bin.2.weight", newT267[:-1].T),
    ("attr_heads.value_bin.2.bias", newT267[-1]),
]:
    save_edit(n267, torch.tensor(np.ascontiguousarray(a267), dtype=torch.float32))

ids267 = sorted(pd.read_csv("/data/tuning.csv", usecols=["stay_id"]).stay_id.unique().tolist())
print("Staged broadened-decoder head, eta .3/.3.")
print("Offline score:", trial263[(0.3, 0.3)]["score"])
print("Independent audit stays:", ids267[100:200])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 267 / broadened-decoder candidate staged\n"
            "Verified current head exactly matches grid242 .5/.4 and its old pair4 rows. "
            "Backed up all seven edits as backup267. Staged arch263 rows and "
            "trial263 .3/.3 final weights/bias; quiet upstream wire unchanged. "
            "Offline NLL deltas +.001399/+.004756/+.004258; immediate slopes "
            "[-1.809,-1.905,+3.126,+2.045]. Pending independent GPU factual audit.\n")
```