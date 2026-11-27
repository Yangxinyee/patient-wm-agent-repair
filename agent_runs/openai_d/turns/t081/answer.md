This confirms the target estimates, but no gate test has run yet. I’ll now check whether the cached inputs reproduce the head and whether reading types can be separated without sensitivity to latent draws. If either check fails, I won’t use the proposed gate.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

r73_snapshot = list(globals().items())
print("=== DATAFRAME CANDIDATES ===")
r73_frames = {}
for r73_n, r73_v in r73_snapshot:
    if isinstance(r73_v, pd.DataFrame) and not r73_n.startswith(("r7", "rec")):
        r73_frames[r73_n] = r73_v
        print(r73_n, r73_v.shape, list(r73_v.columns)[:10])

print("\n=== INNOVATION CACHE CANDIDATES ===")
for r73_n, r73_v in r73_snapshot:
    if not isinstance(r73_v, dict) or r73_n.startswith(("r7", "rec")):
        continue
    if {"valid", "rd"}.issubset(r73_v):
        print(r73_n, list(r73_v.keys()))
    for r73_drug in ("insulin", "norepinephrine"):
        r73_sub = r73_v.get(r73_drug)
        if isinstance(r73_sub, dict) and {"valid", "rd"}.issubset(r73_sub):
            print(r73_n, r73_drug, list(r73_sub.keys()))

r73_x = torch.as_tensor(rep_Q68[0, :100])
r73_w = torch.as_tensor(load_original("attr_heads.value_bin.0.weight"))
r73_b = torch.as_tensor(load_original("attr_heads.value_bin.0.bias"))
r73_h = F.gelu(F.linear(r73_x, r73_w, r73_b))
print("\nHidden-feature RMS mismatch:",
      float((r73_h - torch.as_tensor(rep_H68[0, :100])).square().mean().sqrt()))

r73_inv = 1.0 / np.asarray(load_original("query_norm.weight"))
r73_inv /= np.linalg.norm(r73_inv)
r73_qstat = (rep_Q68 - np.asarray(rep_fixed68)[None]) @ r73_inv
r73_stat = rep_Q68 @ r73_inv
print("Query invariant SD:", float(r73_qstat.std()))
print("Maximum within-row latent-draw range:",
      float(np.ptp(r73_stat, axis=0).max()))

for r73_n, r73_df in r73_frames.items():
    if len(r73_df) != rep_Q68.shape[1]:
        continue
    r73_col = next((c for c in ("event", "code", "event_name")
                   if c in r73_df.columns), None)
    if r73_col:
        print("Gate ranges using", r73_n, r73_col)
        print(pd.DataFrame({
            "event": r73_df[r73_col].to_numpy(),
            "gate": r73_stat.mean(axis=0)
        }).groupby("event").gate.agg(["min", "max", "mean", "std"]))

print("Current edits:", edited())
```