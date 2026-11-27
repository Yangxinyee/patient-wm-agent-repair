The model’s local dose responses are much weaker than the record estimates: about −0.47 per insulin unit and +0.07 per norepinephrine unit. The sensitivity matrix agrees with direct finite differences. I’ll fit new drug-separated directions using stay-held-out validation, then check their nonlinear responses and factual perturbations before submitting an edit.

```python
import numpy as np, torch
from scipy.linalg import solve

# Immediate effects estimated from the training records, not medical assumptions.
targets158 = np.array([
    [-1.93474217, -2.01483673, 0., 0.],
    [0., 0., 3.05259229, 2.52816442]
], dtype=np.float64)

stay158 = np.asarray(rr33_stays)[indices151]
unique158 = np.unique(stay158)
rng158 = np.random.default_rng(158)
rng158.shuffle(unique158)
validation_stays158 = set(unique158[::4].tolist())
validation158 = np.array([s in validation_stays158 for s in stay158])
training158 = ~validation158

ratio158 = dz156 / dc156[:, None]
baseline_rate158 = baseline_contrast156 / dc156[:, None]
X158 = (
    J156[:, :, :, None].astype(np.float64)
    * ratio158[:, None, None, :]
).reshape(len(indices151), 4, 384)
Y158 = targets158[drug156] - baseline_rate158

event_scale158 = np.array([2., 2., 3., 3.])
rate_weight158 = np.empty(len(dc156))
for d158 in range(2):
    q158 = drug156 == d158
    rate_weight158[q158] = np.clip(
        dc156[q158] / np.median(dc156[q158]), .25, 3.
    )
weights158 = (
    np.sqrt(rate_weight158)[:, None] / event_scale158[None, :]
)
Xw158 = X158 * weights158[:, :, None]
Yw158 = Y158 * weights158

def solve_routing158(mask, alpha):
    x = Xw158[mask].reshape(-1, 384)
    y = Yw158[mask].reshape(-1)
    gram = x.T @ x / len(x)
    rhs = x.T @ y / len(x)
    return solve(
        gram + alpha*np.eye(384), rhs, assume_a="pos"
    ).reshape(192, 2)

alphas158 = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
             3e-4, 1e-3, 3e-3, 1e-2]
fits158 = {}
scores158 = []
print("training/validation stays:",
      len(np.unique(stay158[training158])),
      len(np.unique(stay158[validation158])))
print("event order:", events151)
for alpha158 in alphas158:
    dtrain158 = solve_routing158(training158, alpha158)
    prediction158 = baseline_rate158 + np.einsum(
        "nek,kc,nc->ne", J156, dtrain158, ratio158
    )
    error158 = (
        prediction158[validation158]
        - targets158[drug156[validation158]]
    ) / event_scale158
    score158 = float(np.sqrt(np.mean(error158**2)))
    dall158 = solve_routing158(
        np.ones(len(indices151), dtype=bool), alpha158
    )
    fits158[alpha158] = dall158
    scores158.append((score158, alpha158))
    print("alpha", alpha158,
          "held-out normalized RMSE", round(score158, 4),
          "full-fit direction norms",
          np.round(np.linalg.norm(dall158, axis=0), 3))

print("\n=== NONLINEAR CHECKS OF THREE BEST FITS ===")
nonlinear158 = {}
for score158, alpha158 in sorted(scores158)[:3]:
    D158 = fits158[alpha158]
    dtensor158 = torch.as_tensor(D158, dtype=torch.float32)
    token_fact158 = token_base154 + z_base154[:, 30:32] @ dtensor158.T
    token_cf158 = tokencf156 + zcf156[:, 30:32] @ dtensor158.T
    of158, hf158 = local_token154(token_fact158)
    oc158, hc158 = local_token154(token_cf158, ucf156)
    vf158 = dynamic_views154(of158, hf158)
    vc158 = dynamic_views154(oc158, hc158)
    nonlinear_rate158 = dynamic_delta154(vc158, vf158) / dc156[:, None]
    factual_shift158 = dynamic_delta154(vf158, views_base154)
    nonlinear158[alpha158] = {
        "rate_response": nonlinear_rate158,
        "factual_shift": factual_shift158,
        "D": D158
    }
    print("alpha", alpha158,
          "factual shift RMS",
          np.round(np.sqrt(np.mean(factual_shift158**2, axis=0)), 3))
    for d158 in range(2):
        q158 = drug156 == d158
        print(" drug", d158,
              "nonlinear response mean",
              np.round(nonlinear_rate158[q158].mean(0), 3),
              "SD", np.round(nonlinear_rate158[q158].std(0), 3))

best_alpha158 = min(scores158)[1]
D_best158 = fits158[best_alpha158]
candidate158 = {k: v.clone() for k, v in zero_weights154.items()}
candidate158["z_in.weight"][:, 30:32] = torch.as_tensor(
    D_best158, dtype=torch.float32
)
print("\nCandidate retained, NOT saved; selected alpha:", best_alpha158)

with open("/work/notes.md", "a") as f158:
    f158.write(
        "\n## Run 158: closed-form drug-specific token routing\n"
        "- Original local responses understate identified effects: insulin/finger about -0.468; NE/arterial about +0.070 per rate unit.\n"
        "- Fit ridge response directions using a numerical exact-core Jacobian, drug-specific physical-dose contrasts, and stay-held-out validation.\n"
        "- Tested nonlinear local responses and factual mean perturbations of the best regularization settings.\n"
        f"- Retained candidate158, alpha={best_alpha158}; not saved. Python runs: 158; forward passes: 10.\n"
    )
```