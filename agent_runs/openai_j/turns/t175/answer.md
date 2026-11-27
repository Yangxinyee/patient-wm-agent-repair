The candidate reproduces the fitted correction closely, and its local tuning NLL increase is 0.00319—inside the 0.005 limit. MAP predictions are essentially unchanged locally. I’ll save these four value-head tensors, then test them on the GPU; a larger factual loss increase would invalidate the local safety estimate.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as _F159
from pathlib import Path

def trial159_mean(mc):
    h = _F159.gelu(_F159.linear(
        mc["inputs"], trial158_W0, trial158_b0
    ))
    new_y = _F159.linear(h, trial158_W2, trial158_b2)
    old_y = _F159.linear(
        mc["hidden"],
        wm_weights145["attr_heads.value_bin.2.weight"],
        wm_weights145["attr_heads.value_bin.2.bias"]
    )
    p = torch.softmax(mc["logits"] + new_y - old_y, -1)
    return (p.mean(0) * wm_train145["centres"]).sum(-1).numpy()

trial159_base_mean = trial159_mean(wm_train145)
trial159_rows = []
for trial159_key, trial159_cf in wm_cf146.items():
    trial159_mu = trial159_mean(trial159_cf)
    trial159_dr = np.asarray(trial159_cf["delta_rate"])
    trial159_active = np.asarray(trial159_cf["active"], dtype=bool)
    for trial159_event in sorted(head_meta.event.unique()):
        trial159_mask = (
            trial159_active
            & (head_meta.event.to_numpy() == trial159_event)
            & (np.abs(trial159_dr) > 1e-8)
        )
        if trial159_mask.any():
            trial159_shift = trial159_mu[trial159_mask] - trial159_base_mean[trial159_mask]
            trial159_rows.append({
                "drug": trial159_key[0], "factor": trial159_key[1],
                "sensor": trial159_event, "n": int(trial159_mask.sum()),
                "mean_shift": float(trial159_shift.mean()),
                "slope": float(
                    trial159_shift.sum() / trial159_dr[trial159_mask].sum()
                )
            })
print(pd.DataFrame(trial159_rows).to_string(index=False))

for trial159_name, trial159_tensor in trial158_weights.items():
    save_edit(trial159_name, trial159_tensor)
print("\nSaved edits:", edited())
print("Candidate: glucose-only head correction, gain 1.4; all other heads/core unchanged.")
with Path("/work/notes.md").open("a") as trial159_f:
    trial159_f.write(
        "\n## Run 159: installed glucose-only stochastic candidate\n"
        "- Saved four value-head tensors from run 158; gain 1.4.\n"
        "- Other heads, embeddings, recurrent core and transition tables are unchanged.\n"
        "- Local tuning value dNLL +0.003192; GPU factual check required next.\n"
        "- Four active edits; forward passes used: 14.\n"
        + pd.DataFrame(trial159_rows).to_string(index=False) + "\n"
    )
```
