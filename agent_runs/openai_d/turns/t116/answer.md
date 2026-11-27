The local correction is effectively silent outside active-insulin glucose readings. Held-out factual NLL rises by 0.00237 nats, below the 0.005 limit, and replacing the two neurons alone has negligible cost. I’ll install the full-strength candidate and then check it with the GPU tool. A larger GPU NLL increase would refute the local validation.

```python
import torch, numpy as np, pandas as pd
from pathlib import Path

assert edited() == [], "Unexpected existing edits; stop before installation."
lambda105 = 1.0
trial105 = {
    "attr_heads.value_bin.0.weight": W1103.clone(),
    "attr_heads.value_bin.0.bias": b1103.clone(),
    "attr_heads.value_bin.2.weight": W2103.clone(),
    "attr_heads.value_bin.2.bias": b2103.clone()
}
for j105, sign105 in zip(pair104, (1.0, -1.0)):
    trial105["attr_heads.value_bin.0.weight"][j105] = torch.tensor(
        gate_strength104*gv104 + sign105*decoder_step104*dv104/2,
        dtype=torch.float32)
    trial105["attr_heads.value_bin.0.bias"][j105] = float(
        gate_strength104*gb104 + sign105*decoder_step104*db104/2)
    trial105["attr_heads.value_bin.2.weight"][:,j105] = torch.tensor(
        sign105*lambda105*theta104*tilt104/decoder_step104,
        dtype=torch.float32)
trial105["attr_heads.value_bin.2.bias"] += (
    W2103[:,pair104] @ torch.tensor(hbar104, dtype=torch.float32)
)

with torch.no_grad():
    x105 = cache98["with_code"]
    oldraw105 = (
        torch.nn.functional.gelu(x105 @ W1103.T + b1103) @ W2103.T + b2103
    )
    newraw105 = (
        torch.nn.functional.gelu(
            x105 @ trial105["attr_heads.value_bin.0.weight"].T
            + trial105["attr_heads.value_bin.0.bias"]
        ) @ trial105["attr_heads.value_bin.2.weight"].T
        + trial105["attr_heads.value_bin.2.bias"]
    )
    delta105 = (newraw105-oldraw105).cpu().numpy().astype(np.float64)

expected105 = -removed104 + lambda105*theta104*R104[:,:,None]*tilt104
err105 = float(np.max(np.abs(delta105-expected105)))
print("Installed-tensor versus intended logit delta max error:", err105)
assert err105 < 0.003

L105 = baseL104 + delta105
L105 -= L105.max(axis=-1,keepdims=True)
P105 = np.exp(L105)
P105 /= P105.sum(axis=-1,keepdims=True)
pm105 = P105.mean(axis=0)
nn105 = -np.log(np.maximum(pm105[np.arange(len(bin104)),bin104],1e-35))
print("Exact candidate ΔNLL, all/train/held-out:",
      float((nn105-baseNLL104).mean()),
      float((nn105-baseNLL104)[trainmask99].mean()),
      float((nn105-baseNLL104)[valmask99].mean()))

for name105, tensor105 in trial105.items():
    assert tuple(tensor105.shape) == tuple(SHAPES[name105])
    assert torch.isfinite(tensor105).all()
    save_edit(name105, tensor105)

tuning_ids105 = sorted(pd.read_csv("/work/fwd_1/factual_original.csv").stay_id.unique().tolist())
print("Tuning IDs: count, first, last, contiguous:",
      len(tuning_ids105), tuning_ids105[0], tuning_ids105[-1],
      tuning_ids105 == list(range(tuning_ids105[0],tuning_ids105[-1]+1)))
print("CURRENT EDITS:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 105: installed insulin candidate, awaiting GPU validation\n"
            "Raw-coordinate conversion verified. Gate inactive RMS=1.26e-6; "
            "active decoder reconstruction RMSE=2.88e-5. Local full-strength "
            "held-out ΔNLL=+0.002366; replacement-only ΔNLL=-0.000018. "
            "Installed symmetric GELU innovation gate in value-head neurons "
            "107/109 with mean-contribution compensation. Four value-head "
            "tensors edited; all other tensors original. Must validate paired "
            "tuning factual NLL, then dose rollouts. Candidate is provisional.\n")
```
