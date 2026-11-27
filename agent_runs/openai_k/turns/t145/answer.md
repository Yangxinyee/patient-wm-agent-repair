The earlier scale discrepancy came from comparing physical rates with dose innovations. In physical units, insulin decoding is close to linear; norepinephrine still bends at lower doses. The failed candidate changed factual means only slightly, suggesting a weak or poorly routed state-to-value correction. I’ll test whether its modified states retain enough drug information for a more direct, centered readout correction.

```python
import numpy as np
import torch

print("=== INNOVATION TARGET CHECK ===")
for r134_d, r134_drug in enumerate(["insulin", "norepinephrine"]):
    r134_active = r133_physical[:, r134_d] > 0
    print(r134_drug,
          "retained innovation mean/SD on active bundles:",
          np.mean(rtrue66[r134_active, r134_d]),
          np.std(rtrue66[r134_active, r134_d]),
          "non-action nonzero count:",
          np.count_nonzero(np.abs(rtrue66[~r134_active, r134_d]) > 1e-6))

r134_index = np.asarray(index66, dtype=int)
r134_x = torch.as_tensor(X66, dtype=torch.float32)
r134_original_u = torch.as_tensor(rr33_u_all[r134_index])
r134_base_mask = (
    (r134_x[:, 384:] - r134_original_u).abs().amax(dim=1).numpy() < 1e-5)
print("augmentation rows / factual rows:", len(r134_index), r134_base_mask.sum())
for r134_n in ["dr66", "delta88"]:
    r134_v = np.asarray(globals()[r134_n])
    print(r134_n,
          "factual-row RMSE against physical rate:",
          np.sqrt(np.mean(
              (r134_v[r134_base_mask] -
               rate66[r134_index[r134_base_mask]])**2, axis=0)),
          "against innovation:",
          np.sqrt(np.mean(
              (r134_v[r134_base_mask] -
               rtrue66[r134_index[r134_base_mask]])**2, axis=0)))

print("\n=== AUGMENTATION EXAMPLES ===")
for r134_i in np.flatnonzero(np.any(r133_physical > 0, axis=1))[:3]:
    r134_rows = np.flatnonzero(r134_index == r134_i)[:6]
    print("state", r134_i, "physical", r133_physical[r134_i],
          "innovation", rtrue66[r134_i])
    print("rows", r134_rows,
          "dr66", np.round(np.asarray(dr66)[r134_rows], 4),
          "delta88", np.round(np.asarray(delta88)[r134_rows], 4))

print("\n=== PRIOR-FEATURE INTERFACE TEST ===")
r134_saved = {k: W0[k] for k in candidates90[1.0]}
try:
    for r134_k, r134_v in candidates90[1.0].items():
        W0[r134_k] = r134_v.detach().clone()
    r134_probe = r132_ix[:12]
    r134_pf = prior_features74(r134_probe)
    r134_pfcf = prior_features74(
        r134_probe,
        u=rr35_scaled_action_encoding(r134_probe, "insulin", 2.0))
    for r134_label, r134_obj in [
        ("factual", r134_pf), ("insulin doubled", r134_pfcf)]:
        if isinstance(r134_obj, (tuple, list)):
            print(r134_label, type(r134_obj).__name__,
                  [tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                   for v in r134_obj])
        else:
            print(r134_label, type(r134_obj).__name__,
                  getattr(r134_obj, "shape", None))
    if isinstance(r134_pf, np.ndarray) and isinstance(r134_pfcf, np.ndarray):
        print("feature-change mean norm:",
              np.linalg.norm(r134_pfcf - r134_pf, axis=1).mean())
finally:
    for r134_k, r134_v in r134_saved.items():
        W0[r134_k] = r134_v

print("\n=== PRIOR SENSITIVITY SCALE ===")
print("J65 mean sensitivity norms by sensor:",
      np.mean(np.linalg.norm(J65, axis=2), axis=0))
for r134_penalty in [0.001, 0.01, 0.1, 1.0]:
    print(r134_penalty, "prior directions:",
          np.linalg.norm(D65_candidates[r134_penalty], axis=0))

with open("/work/notes.md", "a") as r134_file:
    r134_file.write(
        "\n## Run 134: test direct prior-head correction route\n"
        "- Corrected dose audit: insulin channel change is approximately 1:1 with physical rate; NE is asymmetric (x0 .65, half .86, x2 1.13).\n"
        "- rtrue66 is a centered innovation target, not a physical-rate target.\n"
        "- Testing whether modified-state prior features can decode treatment innovations for a stronger, targeted readout correction.\n"
        "- Python runs: 134; forward passes: 6; no submitted edits.\n"
    )
```