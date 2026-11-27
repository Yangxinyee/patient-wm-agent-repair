The inspection stopped on a NumPy/Torch type mismatch; it changed no weights. I’ll fix that and measure the latent-to-reading sensitivities with forward-only perturbations. This will give us a proposed edit direction, not a causal estimate. A stable insulin direction with little MAP spillover—and the reverse for norepinephrine—would support testing the edit on the GPU.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

torch.set_num_threads(8)

def np45(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def tt45(x):
    return x.float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)

core_config45 = (4, 1.0, "both", True)
hbase45, obase45 = core_update_local(tt45(core_z), core_config45)
print("Core baseline errors:",
      float(torch.mean((hbase45 - tt45(core_target_h))**2).sqrt()),
      float(torch.mean((obase45 - tt45(core_target_out))**2).sqrt()))
print("Target h/out RMS:",
      float(tt45(core_target_h).square().mean().sqrt()),
      float(tt45(core_target_out).square().mean().sqrt()))

all_out45 = np45(core_state)[:, :192]
target_out45 = np45(core_target_out)
actual_rows45 = np.array([
    np.argmin(np.sum((all_out45 - row)**2, axis=1)) for row in target_out45
])
meta_by_state45 = state_df.set_index("state_row")
selected_meta45 = meta_by_state45.loc[actual_rows45].reset_index()
links45 = []
for j45, row45 in selected_meta45.iterrows():
    for event45 in ["glucose_finger", "map_arterial"]:
        candidates45 = head_meta[
            (head_meta.stay_id == row45.stay_id) &
            (head_meta.event == event45) &
            (head_meta.hour > row45.hour + 1e-5) &
            (head_meta.hour <= row45.hour + 1.1)
        ]
        if len(candidates45):
            k45 = int(candidates45.sort_values("hour").index[0])
            links45.append((j45, k45, event45,
                            (float(head_meta.loc[k45, "hour"]) - float(row45.hour))*3600))
link_df45 = pd.DataFrame(links45, columns=["core_index", "head_index", "event", "dt"])
print("Linked next readings:", link_df45.event.value_counts().to_dict())
print("Posterior reconstruction function result:",
      type(_post33(cf_rows[:2], tt45(cf_E[:2]), tt45(cf_U[:2]))).__name__)

src45 = torch.tensor(link_df45.core_index.to_numpy(), dtype=torch.long)
idx45 = torch.tensor(link_df45.head_index.to_numpy(), dtype=torch.long)
dt45 = torch.tensor(link_df45.dt.to_numpy(), dtype=torch.float32)[:, None]
static45 = tt45(head_static)[idx45]
trans45 = tt45(head_transition)[idx45]
centres45 = tt45(head_centres)[idx45]

def next_means45(h45, out45):
    flowed45 = h45[src45] * torch.exp(
        -dt45[:, :, None] * torch.exp(W0["core.flow.a"])[None, None, :]
    )
    ctx45 = out45[src45] + norm0(flowed45.mean(dim=1), "core.out_norm")
    ctx45 = ctx45 + mlp0(torch.log(dt45.clamp_min(1e-6))/10.0, "gap")
    prior45 = mlp0(ctx45, "z_prior")[:, :32]
    query45 = norm0(
        ctx45 + linear0(prior45, "z_head") + W0["type_emb.weight"][1],
        "query_norm"
    )
    hidden45 = F.gelu(linear0(query45 + static45, "attr_heads.value_bin.0"))
    logits45 = linear0(hidden45, "attr_heads.value_bin.2") + trans45
    return (torch.softmax(logits45, dim=-1) * centres45).sum(dim=-1)

baseline_means45 = next_means45(hbase45, obase45)
jac45 = np.zeros((len(link_df45), 32), dtype=np.float64)
eps45 = 0.10
for dim45 in range(32):
    perturb45 = torch.zeros_like(tt45(core_z))
    perturb45[:, dim45] = eps45
    hp45, op45 = core_update_local(tt45(core_z) + perturb45, core_config45)
    hm45, om45 = core_update_local(tt45(core_z) - perturb45, core_config45)
    jac45[:, dim45] = np45(
        (next_means45(hp45, op45) - next_means45(hm45, om45))/(2*eps45)
    )

latent_directions45 = {}
for drug45, desired45 in [
    ("insulin", np.array([-1.0, 0.0])),
    ("norepinephrine", np.array([0.0, 1.0]))
]:
    relevant45 = selected_meta45["has_" + drug45].to_numpy()[link_df45.core_index]
    rows45 = []
    for event45 in ["glucose_finger", "map_arterial"]:
        use45 = relevant45 & (link_df45.event.to_numpy() == event45)
        if not use45.any():
            use45 = link_df45.event.to_numpy() == event45
        rows45.append(jac45[use45].mean(axis=0))
    J45 = np.stack(rows45)
    direction45 = np.linalg.lstsq(J45, desired45, rcond=1e-3)[0]
    latent_directions45[drug45] = direction45.astype(np.float32)
    print(drug45, "Jacobian singular values", np.linalg.svd(J45, compute_uv=False),
          "direction norm", np.linalg.norm(direction45),
          "predicted glucose/MAP response", J45 @ direction45)

print("Feature standardisation check:",
      np45(cf_F0)[0, :4], np45(cf_H0)[0, :4]/np45(cf_scale)[:4])
for key45, coeff45 in cf_fits.items():
    factual45 = np45(cf_F0) @ coeff45
    other45 = np45(cf_Fother) @ coeff45
    print("feature", key45,
          "factual RMS", np.sqrt(np.mean(factual45**2, axis=0)),
          "nonaction RMS", np.sqrt(np.mean(other45**2, axis=0)))
for name45, mapping45 in [("starts", cf_starts), ("ends", cf_ends)]:
    sample45 = next(iter(mapping45.values()))
    print(name45, type(sample45).__name__,
          type(sample45[0]).__name__ if isinstance(sample45, (list, tuple)) and len(sample45) else "")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 45: forward-only latent sensitivity\n"
            "Computed central-difference posterior-latent sensitivity to next glucose and MAP readings, "
            "using the approximate local core only as an edit-direction diagnostic. "
            "No gradients or weight edits. Python runs: 45; forward passes: 3.\n")
```
