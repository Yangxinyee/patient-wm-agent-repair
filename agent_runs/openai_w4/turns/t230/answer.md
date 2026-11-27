The record estimates support positive NE effects, while the local response calibration produces negative rollout effects. I’ll test the reversed connection without changing submitted weights. If it preserves factual fit, a paired rollout can determine whether reversing this direction actually repairs the sustained response.

```python
import numpy as np
import torch

torch.set_num_threads(8)
current202 = {n: torch.as_tensor(load(n)).detach().clone() for n in NAMES}
old_rW202 = rW
old_bank202 = repair_bank
old_linear202 = rp_linear

dual_b202 = torch.as_tensor(backup_dual198["core.b_in.weight"]).detach().clone()
unit_ne202 = 2.0 * (dual_b202 - current202["core.b_in.weight"])
base_b202 = current202["core.b_in.weight"] - 0.5 * unit_ne202

print("NE CONNECTION SINGULAR VALUES",
      np.linalg.svd(as_np184(unit_ne202), compute_uv=False)[:5])
print("OTHER CHANNEL DIFFERENCES FROM SCALE-1 BACKUP",
      {n: float(torch.max(torch.abs(current202[n] -
              torch.as_tensor(backup_dual198[n]))))
       for n in backup_dual198
       if n != "core.b_in.weight" and n in current202})

def small_result202(x):
    if isinstance(x, dict):
        return {k: small_result202(v) for k, v in x.items()
                if k not in ("losses", "states", "predictions", "means")}
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        a = as_np184(x)
        return a.tolist() if a.size <= 8 else {"shape": list(a.shape), "mean": float(np.nanmean(a))}
    if isinstance(x, (tuple, list)):
        return [small_result202(v) for v in x[:8]]
    return x

results202 = {}
try:
    for scale202 in [0.5, 0.0, -0.5, -1.0]:
        candidate202 = dict(current202)
        candidate202["core.b_in.weight"] = base_b202 + scale202 * unit_ne202
        results202[scale202] = factual_eval194(candidate202)
        print("NE SCALE", scale202, small_result202(results202[scale202]))
finally:
    rW = old_rW202
    repair_bank = old_bank202
    rp_linear = old_linear202

print("RECORD KERNELS",
      {k: np.asarray(v).round(4).tolist() for k, v in kernels137.items()})
print("RECENT COMPONENT NAMES",
      " ".join(sorted(n for n in globals()
                      if n.endswith(("193", "194", "199"))
                      and not n.startswith("__"))))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 202\nCompared factual replay at NE connection scales +.5, 0, -.5 and -1, reconstructing the unit response from the scale-1 backup. Submitted weights unchanged; CPU globals restored. Reversed connection is a mechanism test, not evidence about the ICU's drug effect.\n")
```