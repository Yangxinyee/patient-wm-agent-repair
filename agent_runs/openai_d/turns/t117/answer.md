The preflight stopped before saving any edits: floating-point cancellation made the reconstructed logits miss the numerical check by a small amount. I’ll widen the difference between the two gated neurons to reduce cancellation while preserving the intended correction. Matching local NLL and negligible inactive leakage will confirm this adjustment.

```python
import torch, numpy as np, pandas as pd
import torch.nn.functional as F

assert edited() == []
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 105 result\n"
            "Installation aborted before save_edit: max logit reconstruction "
            "error 0.003045 exceeded numerical threshold 0.003. No edits saved. "
            "Likely cancellation between paired gated neurons; increasing "
            "their decoder separation to reduce numerical sensitivity.\n")

decoder_step104 = 0.20
a106 = torch.tensor(gate_strength104*G104, dtype=torch.float32)
d106 = torch.tensor(decoder_step104*D104/2, dtype=torch.float32)
R104 = ((F.gelu(a106+d106)-F.gelu(a106-d106))
        /decoder_step104).numpy()
active_error106 = float(np.sqrt(np.mean(
    (R104[:,active104]-D104[:,active104])**2)))
inactive_rms106 = float(np.sqrt(np.mean(R104[:,~active104]**2)))
print("Active reconstruction RMSE / inactive RMS / inactive max:",
      active_error106, inactive_rms106,
      float(np.max(np.abs(R104[:,~active104]))))
assert active_error106 < 0.005 and inactive_rms106 < 0.001

trial105 = {
    "attr_heads.value_bin.0.weight": W1103.clone(),
    "attr_heads.value_bin.0.bias": b1103.clone(),
    "attr_heads.value_bin.2.weight": W2103.clone(),
    "attr_heads.value_bin.2.bias": b2103.clone()
}
for j106, sign106 in zip(pair104, (1.0, -1.0)):
    trial105["attr_heads.value_bin.0.weight"][j106] = torch.tensor(
        gate_strength104*gv104 + sign106*decoder_step104*dv104/2,
        dtype=torch.float32)
    trial105["attr_heads.value_bin.0.bias"][j106] = float(
        gate_strength104*gb104 + sign106*decoder_step104*db104/2)
    trial105["attr_heads.value_bin.2.weight"][:,j106] = torch.tensor(
        sign106*theta104*tilt104/decoder_step104, dtype=torch.float32)
trial105["attr_heads.value_bin.2.bias"] += (
    W2103[:,pair104] @ torch.tensor(hbar104, dtype=torch.float32)
)

with torch.no_grad():
    x106 = cache98["with_code"]
    oldraw106 = F.gelu(x106 @ W1103.T+b1103) @ W2103.T+b2103
    newraw106 = F.gelu(
        x106 @ trial105["attr_heads.value_bin.0.weight"].T
        + trial105["attr_heads.value_bin.0.bias"]
    ) @ trial105["attr_heads.value_bin.2.weight"].T + trial105["attr_heads.value_bin.2.bias"]
    delta106 = (newraw106-oldraw106).numpy().astype(np.float64)

expected106 = -removed104 + theta104*R104[:,:,None]*tilt104
print("Exact logit reconstruction max/RMS error:",
      float(np.max(np.abs(delta106-expected106))),
      float(np.sqrt(np.mean((delta106-expected106)**2))))
assert np.max(np.abs(delta106-expected106)) < 0.003

L106 = baseL104+delta106
L106 -= L106.max(axis=-1,keepdims=True)
P106 = np.exp(L106)
P106 /= P106.sum(axis=-1,keepdims=True)
pm106 = P106.mean(axis=0)
nn106 = -np.log(np.maximum(pm106[np.arange(len(bin104)),bin104],1e-35))
dnn106 = nn106-baseNLL104
print("Exact candidate ΔNLL, all/train/held-out:",
      float(dnn106.mean()), float(dnn106[trainmask99].mean()),
      float(dnn106[valmask99].mean()))
assert dnn106[valmask99].mean() < 0.004

for name106, tensor106 in trial105.items():
    assert tuple(tensor106.shape) == tuple(SHAPES[name106])
    assert torch.isfinite(tensor106).all()
    save_edit(name106, tensor106)

tuning_ids105 = sorted(pd.read_csv(
    "/work/fwd_1/factual_original.csv").stay_id.unique().tolist())
print("Tuning IDs count/first/last/contiguous:",
      len(tuning_ids105), tuning_ids105[0], tuning_ids105[-1],
      tuning_ids105 == list(range(tuning_ids105[0],tuning_ids105[-1]+1)))
print("CURRENT EDITS:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 106: insulin candidate installed\n"
            "Increased symmetric decoder separation from 0.05 to 0.20 "
            "to reduce cancellation. Local checks passed; full-strength "
            "gated correction installed in value-head neurons 107/109, "
            "with mean compensation. Four value-head tensors edited. "
            "Awaiting paired GPU factual validation; not yet accepted.\n")
```
