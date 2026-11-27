The original rollouts are much closer to observed baselines; the reversed candidate drifts by roughly −13 glucose units and +12 MAP units. I’ll test lower-norm response directions using the same dose channel. Smaller factual/state disruption would favor them, but rollout tests must still establish their signs and magnitudes.

```python
import numpy as np
import torch

J206 = np.asarray(J187, dtype=np.float64)
target_ne206 = np.asarray(target_ne187, dtype=np.float64).reshape(-1)
assert J206.shape[0] == len(target_ne206)

ne_directions206 = {
    1.0: np.asarray(directions_ne187[1.0], dtype=np.float64),
    10.0: np.linalg.solve(J206.T @ J206 + 10.0*np.eye(192),
                          J206.T @ target_ne206)
}
print("DIRECTION NORMS",
      {r: float(np.linalg.norm(d)) for r, d in ne_directions206.items()})
print("BASE CONNECTION CHECK",
      float(np.max(np.abs(as_np184(preserved_b194) - as_np184(base_b202)))))

ins_unit206 = np.outer(as_np184(ins_direction187).reshape(-1),
                       np.asarray(joint194["ins"]).reshape(-1))
ins_diff206 = as_np184(preserved_b194) - as_np184(load_original("core.b_in.weight"))
ins_strength206 = float(np.sum(ins_diff206*ins_unit206) / np.sum(ins_unit206**2))
print("INSULIN DECOMPOSITION", {
    "strength": ins_strength206,
    "residual_norm": float(np.linalg.norm(ins_diff206-ins_strength206*ins_unit206)),
    "unit_norm": float(np.linalg.norm(ins_unit206))
})

weights206 = {n: torch.as_tensor(load(n)).detach().clone() for n in NAMES}
ne_decoder206 = torch.as_tensor(joint194["ne"], dtype=torch.float32)
results206 = {}
old_rW206, old_bank206, old_linear206 = rW, repair_bank, rp_linear
try:
    for ridge206, scale206 in [
        (1.0, .5), (1.0, 1.0), (1.0, 2.0),
        (1.0, -.5), (1.0, -1.0), (1.0, -2.0),
        (10.0, 1.0), (10.0, 2.0), (10.0, 4.0), (10.0, -2.0)
    ]:
        b206 = torch.as_tensor(preserved_b194).clone()
        b206 += scale206 * torch.outer(
            torch.as_tensor(ne_directions206[ridge206], dtype=torch.float32),
            ne_decoder206)
        cand206 = dict(weights206)
        cand206["core.b_in.weight"] = b206
        result206 = factual_eval194(cand206)
        results206[(ridge206, scale206)] = result206
        print("NE CANDIDATE", (ridge206, scale206), {
            "dNLL": float(np.mean(result206[1]) - original_mean203),
            "state_rmse": float(np.sqrt(np.mean(
                (result206[0] - reference_x125)**2)))
        })
finally:
    rW, repair_bank, rp_linear = old_rW206, old_bank206, old_linear206

print("INSULIN TARGET OBJECT",
      small_result202(old_targets187))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 206\nMatched observed baselines show reversed NE candidate has substantial free-rollout drift (glucose bias -13.0, MAP +11.9), while original biases are -3.94/+1.58. Tested lower-norm NE response directions at ridge 1 and 10 with factual replay, keeping validated dual dose channel and insulin response fixed. Recovered insulin rank-one decomposition. No submitted weights changed.\n")
```