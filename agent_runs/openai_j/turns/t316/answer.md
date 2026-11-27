The fit preserves factual atom outputs to about \(10^{-5}\), but its weight change is large. I’ll check preservation through the complete action encoder, then install the regularized candidate with a backup. A paired rollout must improve norepinephrine doubling without materially disturbing the other readouts; otherwise I’ll restore it.

```python
from pathlib import Path
import numpy as np
import torch

action_ridge276 = 10.0
action_scale276 = 1.0
action_beta276 = action_scale276 * atom_extrap_betas273[action_ridge276]
action_names276 = (
    "encode_action.mlp1.2.weight",
    "encode_action.mlp1.2.bias",
)
action_candidate276 = {
    action_names276[0]: (
        np.asarray(load_original(action_names276[0]), dtype=np.float64)
        + action_beta276[:-1].T
    ).astype(np.float32),
    action_names276[1]: (
        np.asarray(load_original(action_names276[1]), dtype=np.float64)
        + action_beta276[-1]
    ).astype(np.float32),
}
assert all(np.isfinite(x).all() for x in action_candidate276.values())

u_factual_original276 = repair_set_encode(
    post_av0, post_action_mask, "encode_action", True
)
u_double_original276 = repair_changed_action("norepinephrine", 2.0)
assert isinstance(u_double_original276, torch.Tensor)

saved_wt276 = {name: wt[name] for name in action_names276}
try:
    for name in action_names276:
        wt[name] = torch.as_tensor(action_candidate276[name])
    u_factual_candidate276 = repair_set_encode(
        post_av0, post_action_mask, "encode_action", True
    )
    u_double_candidate276 = repair_changed_action("norepinephrine", 2.0)
finally:
    for name, tensor in saved_wt276.items():
        wt[name] = tensor

u_factual_error276 = float(
    (u_factual_candidate276 - u_factual_original276).abs().max()
)
print("Complete action-encoder factual max error:", u_factual_error276)
assert u_factual_error276 < 5e-4

active276 = torch.as_tensor(repair_has_action, dtype=torch.bool)
old_contrast276 = (
    u_double_original276 - u_factual_original276
)[active276]
new_contrast276 = (
    u_double_candidate276 - u_factual_candidate276
)[active276]
print("Original doubled-dose action contrast mean norm:",
      float(old_contrast276.norm(dim=1).mean()))
print("Candidate doubled-dose action contrast mean norm:",
      float(new_contrast276.norm(dim=1).mean()))
print("Candidate action vectors finite:",
      bool(torch.isfinite(u_double_candidate276).all()))
assert torch.isfinite(u_double_candidate276).all()

np.savez(
    "/work/pre_action_extrap_backup276.npz",
    **{name: np.asarray(load(name), dtype=np.float32) for name in edited()}
)
for name, tensor in action_candidate276.items():
    save_edit(name, tensor)

np.savez(
    "/work/action_extrap_candidate276.npz",
    **{name: np.asarray(load(name), dtype=np.float32) for name in edited()}
)
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 276: regularized action extrapolation installed\n"
        "- Closed-form nullspace correction to encode_action.mlp1.2 weight "
        "and bias; ridge 10, scale 1.\n"
        "- Preserves all training-supported start atoms, cached END atoms, "
        "and within-support intervention grids.\n"
        f"- Complete cached factual action-encoder max error "
        f"{u_factual_error276:.8g}.\n"
        "- Existing calibrated value head unchanged; prior four-tensor edit "
        "saved in pre_action_extrap_backup276.npz.\n"
        "- Linear extrapolation is a modeling assumption, not newly "
        "identified intervention truth. GPU rollout validation pending.\n"
        "- Counts: 276 Python runs, 34 forward passes.\n"
    )
print("INSTALLED EDITS:", edited())
```