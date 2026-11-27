The pooled projections fail an important check: stopping a drug often produces the same-sign marker as doubling it. They are detecting embedding changes, not dose response. I’ll instead test an attention head that reads the latent token specifically, excluding the changing action token. Correct sign reversal for halving and stopping, with little cross-drug leakage, would support that route.

```python
import numpy as np
from scipy.special import softmax
from pathlib import Path

nt240 = array238(norm215(base238)).astype(np.float64)
tr240, te240 = tr239, te239

# Constant slot query, fitted algebraically on factual training states.
Xq240 = nt240[tr240, :4].reshape(-1, 192)
qvec240 = np.linalg.solve(
    Xq240.T @ Xq240 / len(Xq240) + 1e-5*np.eye(192),
    Xq240.mean(axis=0)
)
q240 = nt240[:, :4] @ qvec240
print("held-out query quantiles",
      np.quantile(q240[te240], [.01, .5, .99]).round(4))

# Linear token-type discriminator: latent token versus all other tokens.
Xk240 = nt240[tr240].reshape(-1, 192)
yk240 = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.]), tr240.sum())
wk240 = np.tile(np.array([1, 1, 1, 1, 1, 1, 6.]), tr240.sum())
kvec240 = np.linalg.solve(
    (Xk240.T * wk240) @ Xk240 / wk240.sum() + 1e-4*np.eye(192),
    Xk240.T @ (wk240 * yk240) / wk240.sum()
)
scores240 = nt240 @ kvec240
print("latent key quantiles",
      np.quantile(scores240[te240, 6], [.01, .5, .99]).round(4))
print("other-token maximum quantiles",
      np.quantile(scores240[te240, :6].max(axis=1), [.01, .5, .99]).round(4))

# Exact latent-coordinate projections through the existing layer norm.
const240 = [x for x in norm215.__code__.co_consts if isinstance(x, str)]
normw240 = next(x for x in const240 if x.endswith(".weight"))
normb240 = next(x for x in const240 if x.endswith(".bias"))
gamma240 = array238(load_original(normw240)).astype(np.float64)
beta240 = array238(load_original(normb240)).astype(np.float64)
zin240 = safe236["z_in.weight"].astype(np.float64)
zb240 = safe236["z_in.bias"].astype(np.float64)
center240 = lambda x: x - x.mean(axis=0, keepdims=True)
A240 = np.column_stack([
    gamma240[:, None] * center240(zin240),
    gamma240 * (zb240-zb240.mean()),
    beta240
])
T240 = np.zeros((34, 2))
T240[16, 0], T240[9, 1] = 1., 1.
T240[32] = -zo238[tr240][:, [16, 9]].mean(axis=0)
V240 = np.linalg.lstsq(A240.T, T240, rcond=1e-8)[0]
print("latent projection constraint error",
      float(np.max(np.abs(A240.T @ V240 - T240))))

selector240 = {}
for temperature in [8., 16., 32.]:
    weights = softmax(
        temperature*q240[:, :, None]*scores240[:, None, :], axis=-1
    )
    pool = np.einsum("nsk,nkd->nsd", weights, nt240).mean(axis=1)
    f = pool @ V240
    selector240[temperature] = {"weights": weights, "pool": pool, "features": f}
    print("\ntemperature", temperature, "factual latent attention quantiles",
          np.quantile(weights[te240, :, 6], [.01, .5, .99]).round(4),
          "marker RMS", np.sqrt(np.mean(f[te240]**2, axis=0)).round(4))
    for drug in ["insulin", "norepinephrine"]:
        result = []
        for mult in [0., .5, 1.5, 2., 3.]:
            c = cf239[(drug, mult)]
            nt = nt240.copy()
            nt[:, 5] = array238(norm215(array238(rp_cf_input[(drug, mult)])[:, 384:576]))
            nt[:, 6] = array238(norm215(c["token"]))
            weights_cf = softmax(
                temperature*q240[:, :, None]*(nt @ kvec240)[:, None, :], axis=-1
            )
            pool_cf = np.einsum("nsk,nkd->nsd", weights_cf, nt).mean(axis=1)
            m = c["mask"] & te240
            result.append(np.round(((pool_cf-pool) @ V240)[m].mean(axis=0), 3).tolist())
        print(drug, result)

print("\nCOMPENSATION OBJECTS")
for k, v in relay206.items():
    print(k, describe236(v))
for name in ["B221", "B227"]:
    if name in globals():
        a = array238(globals()[name])
        print(name, a.shape, "norms",
              np.linalg.norm(a, axis=0).round(3).tolist() if a.ndim == 2 else "")

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 240\n"
        "Rejected pooled markers: stopping/halving frequently had wrong-sign "
        "or cross-drug responses. Tested a closed-form constant-query/token-type "
        "key selector and exact latent-coordinate value projections, retaining "
        "the safe z_in mapping. No weights changed. "
        "Counts: 240 Python runs, 23 forwards.\n"
    )
```