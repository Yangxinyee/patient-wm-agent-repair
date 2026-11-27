The candidates differ only in the graft’s output strength, so they can be scaled exactly without changing its decoder or gate. I’ll make a reversible stronger trial and check its CPU cost first. GPU validation will decide whether the larger insulin response can be obtained within the factual-fit limit.

```python
import torch
import torch.nn.functional as F
import numpy as np

graft_checkpoint_beta2 = {n: torch.as_tensor(load(n)).clone() for n in graft_names}
original_head_tensors = {
    n: torch.as_tensor(load_original(n), dtype=torch.float32) for n in graft_names
}
tuning_x3 = torch.as_tensor(robust_X["tuning"], dtype=torch.float32)
tuning_transition3 = robust_parts["tuning"]["transition"].to(torch.float32)
tuning_bins3 = torch.as_tensor(head_obsbins, dtype=torch.long)
tuning_centres3 = torch.as_tensor(centres_np, dtype=torch.float32)
row_index3 = torch.arange(len(tuning_bins3))

def head_average_probs3(tensors, hidden=None):
    if hidden is None:
        hidden = F.gelu(
            tuning_x3 @ tensors[graft_names[0]].T + tensors[graft_names[1]]
        )
    logits = (hidden @ tensors[graft_names[2]].T
              + tensors[graft_names[3]] + tuning_transition3[None, :, :])
    return logits.softmax(-1).mean(0)

original_probs3 = head_average_probs3(original_head_tensors)
original_nll3 = -original_probs3[row_index3, tuning_bins3].clamp_min(1e-12).log()
original_mean3 = (original_probs3 * tuning_centres3).sum(-1)
zero_hidden3 = F.gelu(
    tuning_x3 @ graft_zero[graft_names[0]].T + graft_zero[graft_names[1]]
)

scaled_candidates3 = {}
scaled_cpu_checks3 = {}
for strength in [-2.0, -8.0, -16.0, -24.0, -32.0]:
    candidate = {n: graft_zero[n] + strength * graft_unit[n] for n in graft_names}
    probs = head_average_probs3(candidate, zero_hidden3)
    nll = -probs[row_index3, tuning_bins3].clamp_min(1e-12).log()
    mean = (probs * tuning_centres3).sum(-1)
    penalty = float((nll - original_nll3).mean())
    scaled_candidates3[strength] = candidate
    scaled_cpu_checks3[strength] = penalty
    print("beta", strength, "CPU NLL penalty", round(penalty, 7))
    for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        mask = torch.as_tensor(events3 == event)
        print(" ", event,
              "penalty", round(float((nll - original_nll3)[mask].mean()), 7),
              "mean shift", round(float((mean - original_mean3)[mask].mean()), 4),
              "shift SD", round(float((mean - original_mean3)[mask].std()), 4))

graft_beta = -24.0
for n, tensor in scaled_candidates3[graft_beta].items():
    save_edit(n, tensor)
print("APPLIED BETA", graft_beta, "EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Stronger reversible insulin trial; Python 77\n")
    f.write("- Robust graft is exactly affine in beta through the two output columns only.\n")
    f.write("- CPU paired penalties: " + repr(scaled_cpu_checks3) + "\n")
    f.write("- Applied beta=-24; beta=-2 checkpoint retained. GPU factual and rollout validation pending.\n")
```
