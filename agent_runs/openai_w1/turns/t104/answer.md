The notes confirm that the trace was designed to encode physical rates and drug presence, including doses outside the factual range. Its old calibration is not trustworthy after the unit correction.

I’ll now rebuild the complete observation/action inputs and check the posterior against cached exported states. Agreement near floating-point precision will confirm that subsequent algebraic edits are based on the model’s actual computation.

```python
import numpy as np
import torch

print("=== ALIGNMENT METADATA ===")
print("state keys:", list(repair_S.keys()))
print("atom/start/owner shapes:",
      tuple(r86_atoms.shape), tuple(r89_start.shape), tuple(r89_owner_all.shape))
print("start dtype/unique:", r89_start.dtype, torch.unique(r89_start).tolist())
print("context difference RMS:",
      float(torch.mean((torch.as_tensor(r36_ctx)-torch.as_tensor(r88_ctx_full))**2).sqrt()))

r95_base_atom = r36_atom
def r95_correct_atom(code, value=np.nan, rate=np.nan, dose=np.nan, end=False):
    result = r95_base_atom(code, value, rate, dose, end).clone()
    if not end and np.isfinite(dose) and code in ("insulin", "norepinephrine"):
        result = result + r86_unit[3 if code == "insulin" else 4]
    return result

try:
    r36_atom = r95_correct_atom
    r95_u = repair_action_encoding()
    r95_action_parts = rr50_action_parts()
finally:
    r36_atom = r95_base_atom

r95_all_atoms = r88_local_atoms({
    "glucose_finger": 0, "glucose_lab": 5,
    "map_arterial": 2, "map_cuff": 2
})
r95_obs_mask = ~r89_start.bool()
r95_n = len(r36_indices)
r95_owners = r89_owner_all.long()
r95_obs_counts = torch.bincount(r95_owners[r95_obs_mask], minlength=r95_n).float()
r95_obs_hidden = r79_mlp(r95_all_atoms[r95_obs_mask], "encode.mlp1")
r95_obs_sums = torch.zeros((r95_n, 192), dtype=torch.float32)
r95_obs_sums.index_add_(0, r95_owners[r95_obs_mask], r95_obs_hidden)
r95_obs_pooled = (
    r95_obs_sums / r95_obs_counts.clamp_min(1).sqrt()[:, None]
    + r79_lin(torch.log1p(r95_obs_counts)[:, None], "encode.size")
)
r95_e = r79_norm(r79_mlp(r95_obs_pooled, "encode.mlp2"), "encode.norm")
r95_ctx = torch.as_tensor(r88_ctx_full, dtype=torch.float32)
r95_post_input = torch.cat([r95_ctx, r95_e, r95_u], dim=1)
r95_post_hidden = torch.nn.functional.gelu(r79_lin(r95_post_input, "z_post.0"))
r95_z = r79_lin(r95_post_hidden, "z_post.2")[:, :32]

print("\n=== REBUILT INPUTS ===")
print("e/u/posterior:", tuple(r95_e.shape), tuple(r95_u.shape), tuple(r95_z.shape))
print("action parts:",
      [(type(v).__name__, getattr(v, "shape", None)) for v in r95_action_parts])

r95_comparisons = []
for r95_name, r95_obj in list(globals().items()):
    if r95_name.startswith("r95"):
        continue
    if isinstance(r95_obj, (np.ndarray, torch.Tensor)) and tuple(r95_obj.shape) == (r95_n, 32):
        r95_candidate = torch.as_tensor(r95_obj, dtype=torch.float32)
        r95_error = float(torch.mean((r95_z-r95_candidate)**2).sqrt())
        r95_comparisons.append((r95_error, r95_name))
print("\n=== POSTERIOR COMPARISONS ===")
for r95_error, r95_name in sorted(r95_comparisons)[:12]:
    print(r95_name, "RMS", r95_error)
if r95_comparisons:
    r95_best_error, r95_best_name = min(r95_comparisons)
    r95_reference = torch.as_tensor(globals()[r95_best_name], dtype=torch.float32).clone()
    print("best per-row RMS quantiles:",
          torch.quantile(((r95_z-r95_reference)**2).mean(1).sqrt(),
                         torch.tensor([0., .5, .9, 1.])).tolist())

with open("/work/notes.md", "a") as r95_f:
    r95_f.write(
        "\n## Run 95\n"
        "Rebuilt complete observation/action encodings with corrected measurement and drug dose units; "
        "checking posterior against retained exported states. No weight edits. "
        "Counts: 95 Python runs, 6 forward passes.\n"
    )
```
