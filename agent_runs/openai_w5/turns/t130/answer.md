The first action-encoder hidden layer retains rates far more accurately than the final layer. That supports building an analytic dose channel there. I’ll also test the factual cost of freeing one prelude attention head, which could carry the channel into the state. If that alone consumes the NLL allowance, I’ll revert it and avoid this design.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from pathlib import Path

rng116 = np.random.default_rng(116)
rate_atoms116, rate_targets116 = [], []
rate_atoms_val116, rate_targets_val116 = [], []
for d, drug in enumerate(["insulin", "norepinephrine"]):
    cap = float(caps105[d])
    q = np.concatenate([np.linspace(0, cap, 2600), rng116.uniform(0, cap, 1600)])
    qv = rng116.uniform(0, cap, 1600)
    a = action_atom_exact(drug, q)
    av = action_atom_exact(drug, qv)
    rate_atoms116.append(a)
    rate_atoms_val116.append(av)
    y = np.zeros((len(q), 4))
    yv = np.zeros((len(qv), 4))
    y[:, d] = q / rate_scale105[d]
    yv[:, d] = qv / rate_scale105[d]
    y[:, 2 + d] = 1
    yv[:, 2 + d] = 1
    rate_targets116.append(y)
    rate_targets_val116.append(yv)

atom116 = torch.cat(rate_atoms116)
atomval116 = torch.cat(rate_atoms_val116)
hidden116 = F.gelu(lin0(atom116, "encode_action.mlp1.0")).numpy()
hiddenval116 = F.gelu(lin0(atomval116, "encode_action.mlp1.0")).numpy()
target116 = np.concatenate(rate_targets116)
targetval116 = np.concatenate(rate_targets_val116)
first_rate116 = Ridge(alpha=0.01).fit(hidden116.astype(np.float64), target116)
error116 = first_rate116.predict(hiddenval116.astype(np.float64)) - targetval116
print("first-hidden held-out RMSE [I, NE/.1, I present, NE present]:",
      np.sqrt(np.mean(error116**2, axis=0)).round(7))
print("max absolute errors:", np.max(np.abs(error116), axis=0).round(6))
print("coefficient norms:", np.linalg.norm(first_rate116.coef_, axis=1).round(3))

prelude_out116 = load_original("core.prelude.0.attn.out.weight").clone()
keep116 = np.ones(192, dtype=bool)
keep116[np.asarray(reserved103)] = False
head_energy116 = np.array([
    float(torch.linalg.vector_norm(prelude_out116[keep116, 48*j:48*(j+1)]))
    for j in range(4)
])
private_head116 = int(np.argmin(head_energy116))
prelude_out116[:, 48*private_head116:48*(private_head116+1)] = 0
save_edit("core.prelude.0.attn.out.weight", prelude_out116)
print("prelude head output norms:", head_energy116.round(4))
print("temporarily freed attention head:", private_head116)
print("edits:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 116: isolated-channel capacity test\n"
            "- Refit rate and drug-presence decoding on original first-hidden action features; no gradients.\n"
            "- Temporarily zeroed one prelude attention-head output (minimum output-weight norm), in addition to the four cleared state coordinates.\n"
            "- This is an ablation diagnostic, not a causal repair. Paired factual testing determines whether its capacity cost is acceptable.\n")
```
