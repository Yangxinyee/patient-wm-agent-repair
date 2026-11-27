The check stopped before changing weights: `wt` is a tensor-loading function, not a dictionary. I’ll wrap it temporarily to test the candidate and restore it afterward. The candidate will be installed only if complete factual action encodings remain effectively unchanged.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("Weight-loader signature:", inspect.signature(wt))
print("Edits before test:", edited())
wt_original277 = wt
action_tensors277 = {
    name: torch.as_tensor(tensor)
    for name, tensor in action_candidate276.items()
}

def candidate_wt277(name, *args, **kwargs):
    if name in action_tensors277:
        return action_tensors277[name]
    return wt_original277(name, *args, **kwargs)

try:
    wt = candidate_wt277
    u_factual_candidate277 = repair_set_encode(
        post_av0, post_action_mask, "encode_action", True
    )
    u_double_candidate277 = repair_changed_action("norepinephrine", 2.0)
finally:
    wt = wt_original277

u_factual_error277 = float(
    (u_factual_candidate277 - u_factual_original276).abs().max()
)
print("Complete action-encoder factual max error:", u_factual_error277)
assert u_factual_error277 < 5e-4
assert torch.isfinite(u_double_candidate277).all()

active277 = torch.as_tensor(repair_has_action, dtype=torch.bool)
old_contrast277 = (
    u_double_original276 - u_factual_original276
)[active277]
new_contrast277 = (
    u_double_candidate277 - u_factual_candidate277
)[active277]
print("Original doubled-dose action contrast mean norm:",
      float(old_contrast277.norm(dim=1).mean()))
print("Candidate doubled-dose action contrast mean norm:",
      float(new_contrast277.norm(dim=1).mean()))

np.savez(
    "/work/pre_action_extrap_backup277.npz",
    **{name: np.asarray(load(name), dtype=np.float32) for name in edited()}
)
for name, tensor in action_candidate276.items():
    save_edit(name, tensor)
np.savez(
    "/work/action_extrap_candidate277.npz",
    **{name: np.asarray(load(name), dtype=np.float32) for name in edited()}
)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Runs 276–277: verified action extrapolation installed\n"
        "- Run 276 stopped before editing because wt is a function.\n"
        "- Run 277 used a temporary loader wrapper, restored in finally.\n"
        "- Installed closed-form action-MLP output correction: ridge 10, "
        "scale 1; existing calibrated value head unchanged.\n"
        f"- Complete factual action-encoder max error "
        f"{u_factual_error277:.8g}.\n"
        "- Previous edit saved in pre_action_extrap_backup277.npz; "
        "new six-tensor edit in action_extrap_candidate277.npz.\n"
        "- GPU factual and paired rollout validation pending.\n"
        "- Counts: 277 Python runs, 34 forward passes.\n"
    )
print("INSTALLED EDITS:", edited())
```