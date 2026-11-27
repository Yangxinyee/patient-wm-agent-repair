The needed state caches survived, and the posterior has a 192-neuron hidden layer that can host a localized edit. Before using it, I need to check which cached encodings reproduce the exposed posterior and how accurately the action vectors recover dose.

Small factual dose-prediction residuals would make a centred edit promising. Large residuals would warn that the gains needed for rollout correction could harm factual fit.

```python
import numpy as np
import torch
from pathlib import Path

print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text()[:2200])

print("\n=== CACHE CONTRACTS ===")
for _n in ("records", "rates152", "e_cal161", "u_cal161", "state_cal161",
           "act_groups152", "obs_groups152", "obs_ends152", "stays156", "hours156"):
    if _n not in globals():
        continue
    _o = globals()[_n]
    print(_n, type(_o).__name__, end=" ")
    if hasattr(_o, "shape") and not callable(_o.shape):
        print(tuple(_o.shape))
        if hasattr(_o, "columns"):
            print("columns:", list(_o.columns))
            print(_o.head(2).to_string(index=False))
    elif isinstance(_o, dict):
        print("length", len(_o), "sample", str(list(_o.items())[:2])[:220])
    elif isinstance(_o, (list, tuple)):
        print("length", len(_o), "sample", str(_o[:2])[:220])
    else:
        print(str(_o)[:220])

print("\n=== ACTION ROW EXAMPLE ===")
print(act_rows152[0])
print("\nCodes:", codes)
print("Action-cache keys:", list(ucache152))

print("\n=== POSTERIOR RECONSTRUCTION ===")
_target165 = torch.as_tensor(np.asarray(S)[:, -32:], dtype=torch.float32)
for _en, _un in (("e152", "u152"), ("e_cal161", "u_cal161")):
    _ee = torch.as_tensor(globals()[_en], dtype=torch.float32)
    _uu = torch.as_tensor(globals()[_un], dtype=torch.float32)
    _xx = torch.cat((ctx152, _ee, _uu), dim=1)
    _pp = mlp136(_xx, "z_post")[:, :32]
    print(_en, _un,
          "posterior RMS error", float(torch.sqrt(torch.mean((_pp-_target165)**2))),
          "target RMS", float(torch.sqrt(torch.mean(_target165**2))),
          "action RMS", float(torch.sqrt(torch.mean(_uu**2))))
print("Context RMS:", float(torch.sqrt(torch.mean(ctx152**2))))

print("\n=== RETAINED ROLLOUT-TARGET NOTES ===")
_notes165 = Path("/work/notes.md").read_text()
for _term in ("convolution", "rollout targets", "policy residual", "innovation"):
    _pos = _notes165.lower().find(_term)
    if _pos >= 0:
        print(_notes165[max(0, _pos-180):_pos+600])

print("\n=== FORWARD ARTIFACTS ===")
for _d in sorted(Path("/work").glob("fwd_*")):
    print(_d.name, [p.name for p in _d.iterdir()])
```
