The smaller candidate changes only the four value-head tensors and has a CPU NLL increase of at most 0.0036 across the cached sets. That makes it worth a GPU test, but not yet acceptable. I’ll install it reversibly and verify the weight construction matches the cached calculation. Then the GPU results can confirm or reject its factual fit and rollout benefit.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

def make_contrast_graft(w, b, scale=1.0):
    out = {name: np.asarray(value, dtype=np.float32).copy()
           for name, value in sc_original.items()}
    n0, n1 = map(int, sc_nodes)
    out[sc_names[0]][n0] = np.asarray(sc_gate_w, np.float32) + np.asarray(w, np.float32)
    out[sc_names[0]][n1] = np.asarray(sc_gate_w, np.float32)
    out[sc_names[1]][n0] = float(sc_gate_b) + float(b)
    out[sc_names[1]][n1] = float(sc_gate_b)
    out[sc_names[2]][:, n0] = float(scale) * np.asarray(sc_rank, np.float32)
    out[sc_names[2]][:, n1] = -float(scale) * np.asarray(sc_rank, np.float32)
    out[sc_names[3]] += np.asarray(sc_bias_comp, np.float32)
    return out

trial_contrast_weights = make_contrast_graft(
    trial_r3["w"], trial_r3["b"], trial_r3["scale"])
xx = torch.from_numpy(sc_fact["tuning"]["x"][:, :80].copy())
old_h = F.gelu(xx @ torch.from_numpy(sc_original[sc_names[0]]).T
               + torch.from_numpy(sc_original[sc_names[1]]))
old_attr = old_h @ torch.from_numpy(sc_original[sc_names[2]]).T \
           + torch.from_numpy(sc_original[sc_names[3]])
removed = old_h[..., sc_nodes] @ torch.from_numpy(sc_original[sc_names[2]][:, sc_nodes]).T
base_attr = old_attr - removed + torch.from_numpy(sc_bias_comp)
new_h = F.gelu(xx @ torch.from_numpy(trial_contrast_weights[sc_names[0]]).T
               + torch.from_numpy(trial_contrast_weights[sc_names[1]]))
new_attr = new_h @ torch.from_numpy(trial_contrast_weights[sc_names[2]]).T \
           + torch.from_numpy(trial_contrast_weights[sc_names[3]])
gg = xx @ torch.from_numpy(np.asarray(sc_gate_w, np.float32)) + float(sc_gate_b)
ss = xx @ torch.from_numpy(np.asarray(trial_r3["w"], np.float32)) + float(trial_r3["b"])
expected = (F.gelu(gg + ss) - F.gelu(gg))[..., None] \
           * torch.from_numpy(sc_rank) * float(trial_r3["scale"])
construction_error = float((new_attr - base_attr - expected).abs().max())
print("Construction max logit error:", construction_error)
assert construction_error < 0.001

print("CPU NLL deltas:")
for name, cache in sc_fact.items():
    result = sc_evaluate(cache, trial_r3["w"], trial_r3["b"], trial_r3["scale"])
    print(name, result["nll"] - cache["original_nll"])

for name, value in trial_contrast_weights.items():
    save_edit(name, torch.from_numpy(value))
print("Installed candidate: lambda=0.1, scale=1; rollback checkpoint retained.")
ids_for_gpu = sorted(pd.read_csv("/work/fwd_8/factual_original.csv")["stay_id"].unique().tolist())
print("GPU stay IDs:", ids_for_gpu)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Reversible protected-contrast trial; Python 178, forwards 11/50\n"
            "- Installed r3 lambda=0.1, scale=1 through four value-head tensors; beta=-2 rollback is preserved.\n"
            f"- Algebraic construction max logit discrepancy={construction_error:.8g}.\n"
            "- Cached CPU NLL deltas: first +0.003596, extra -0.000078, tuning +0.002617.\n"
            "- Candidate is NOT accepted until paired GPU factual and dose-edit tests pass.\n")
```
