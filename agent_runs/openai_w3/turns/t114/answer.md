The direct attention features are no better, and the norepinephrine extractor barely detects halving. I’ll test a more explicit channel: decode dose, subtract a state-predicted factual dose, then carry that innovation through the latent token. This is viable only if dose decoding generalizes and the new attention signal can be isolated from ordinary tokens.

```python
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

torch.set_num_threads(8)
base_u103 = torch.as_tensor(u81, dtype=torch.float32)
base_e103 = torch.as_tensor(e80[idx81], dtype=torch.float32)
base_z103 = posterior_mean(ctx81, base_e103, base_u103)
tokens103 = torch.cat([
    h81 + W["core.slots"][None],
    base_e103[:, None], base_u103[:, None],
    linear(base_z103, "z_in")[:, None]
], dim=1)
norm_tokens103 = layer_norm(tokens103, "core.prelude.0.norm1").numpy()
non103 = norm_tokens103[:, :6].reshape(-1, 192).astype(np.float64)
latent103 = norm_tokens103[:, 6].astype(np.float64)

scenarios103 = [(1., 1.)] + [
    tuple(f if k == j else 1. for k in range(2))
    for j in range(2) for f in [0., .5, 1.5, 2., 3.]
]
U103, R103, G103, NU103 = [], [], [], []
for factors in scenarios103:
    uv = torch.as_tensor(action_vectors87(factors), dtype=torch.float32)
    U103.append(uv.numpy().astype(np.float64))
    R103.append(rate88 * np.asarray(factors)[None])
    G103.append((rate88 > 0).astype(float))
    NU103.append(layer_norm(uv, "core.prelude.0.norm1").numpy().astype(np.float64))
U103 = np.concatenate(U103)
R103 = np.concatenate(R103)
G103 = np.concatenate(G103)
NU103 = np.concatenate(NU103)

# Covariance includes changed action tokens, so novelty cannot simply mean changed dose.
fixed103 = norm_tokens103[:, :5].reshape(-1, 192).astype(np.float64)
mu_non103 = (5*fixed103.mean(axis=0) + NU103.mean(axis=0))/6
second103 = (5*(fixed103.T @ fixed103/len(fixed103)) +
             NU103.T @ NU103/len(NU103))/6
cov_non103 = second103 - np.outer(mu_non103, mu_non103)
evals103, evecs103 = np.linalg.eigh(cov_non103)
print("NONLATENT COVARIANCE smallest eigenvalues",
      np.round(evals103[:16], 9))
difference103 = latent103.mean(axis=0) - mu_non103
gate103 = np.linalg.solve(cov_non103 + 1e-5*np.eye(192), difference103)
gate103 /= difference103 @ gate103
print("LATENT GATE: nonlatent factual SD",
      np.std((non103-mu_non103) @ gate103),
      "changed-action SD", np.std((NU103-mu_non103) @ gate103),
      "latent mean/SD",
      np.mean((latent103-mu_non103) @ gate103),
      np.std((latent103-mu_non103) @ gate103))
print("Original latent raw vector RMS / variation",
      np.sqrt(np.mean(linear(base_z103, "z_in").numpy()**2)),
      np.mean(np.std(linear(base_z103, "z_in").numpy(), axis=0)))

# Decode dose and drug presence using interventions constructed from recorded atoms.
tr_bundle103 = np.asarray(train_fact88).reshape(-1, 4)[:, 0].astype(bool)
tr103 = np.tile(tr_bundle103, len(scenarios103))
XX103 = np.column_stack([U103, np.ones(len(U103))])
YY103 = np.column_stack([R103, G103])
pen103 = 1e-5*np.eye(193)
pen103[-1, -1] = 1e-10
decoder103 = np.linalg.solve(
    XX103[tr103].T @ XX103[tr103] + len(XX103[tr103])*pen103,
    XX103[tr103].T @ YY103[tr103])
pred103 = XX103 @ decoder103
print("\nHELD-OUT DOSE/PRESENCE DECODER RMSE",
      np.sqrt(np.mean((pred103[~tr103]-YY103[~tr103])**2, axis=0)))
for j, drug in enumerate(["insulin", "norepinephrine"]):
    present = G103[:, j] > .5
    print(drug, "presence quantiles, absent/present",
          np.round(np.quantile(pred103[~tr103 & ~present, j+2], [.01,.5,.99]), 3),
          np.round(np.quantile(pred103[~tr103 & present, j+2], [.01,.5,.99]), 3))

# Separate policy fits use only pre-action state, never the observed dose as an input.
CX103 = ctx81.numpy().astype(np.float64)
cm103 = CX103[tr_bundle103].mean(axis=0)
cs103 = np.maximum(CX103[tr_bundle103].std(axis=0), .05)
PX103 = np.column_stack([(CX103-cm103)/cs103, np.ones(len(CX103))])
policy_fits103 = {}
print("\nHELD-OUT FACTUAL POLICY DOSE RMSE")
for j, drug in enumerate(["insulin", "norepinephrine"]):
    active = rate88[:, j] > 0
    trainmask = tr_bundle103 & active
    heldmask = ~tr_bundle103 & active
    for lam in [.001, .01, .1, 1.]:
        penalty = lam*np.eye(193)
        penalty[-1, -1] = 1e-8
        b = np.linalg.solve(
            PX103[trainmask].T @ PX103[trainmask]/trainmask.sum() + penalty,
            PX103[trainmask].T @ rate88[trainmask, j]/trainmask.sum())
        policy_fits103[(j, lam)] = b
        err = PX103[heldmask] @ b - rate88[heldmask, j]
        print(drug, lam, "RMSE", round(float(np.sqrt(np.mean(err**2))), 4),
              "bias", round(float(err.mean()), 4))

print("\nSAVED POSTERIOR PILOTS")
for k, v in sorted(list(globals().items())):
    if "pilot" in k and isinstance(v, dict):
        print(k, {n: np.shape(x) for n, x in v.items()})
print("\nCURRENT EDITS", edited())
```
