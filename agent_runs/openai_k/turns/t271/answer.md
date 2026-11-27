Adding non-drug bundles makes the quiet-feature constraint substantially harder: the fitted features recover only part of the dose contrast. That does not yet rule out an edit, because a small state perturbation may have a useful effect at the readout. Next I’ll measure that response with finite differences and fit separate glucose and MAP directions. Consistent target shifts with little cross-effect would support a cautious official test.

```python
import numpy as np, torch

r244_sel = np.unique(np.linspace(0, len(indices151)-1, 24).astype(int))
r244_ids = np.asarray(indices151)[r244_sel]
r244_pre = base228["pre"][r244_sel].detach().clone()
r244_eps = 0.03

def r244_read_pre(pre, ids):
    h = audit_block(linear228(pre, "core.c_out"), "core.coda.0")
    out = norm0(h.mean(1), "core.out_norm")
    return r49_readouts(out, h, ids)

# Numerical response to pre-coda perturbations; no autograd or weight optimisation.
r244_J = np.zeros((len(r244_sel), 4, 192), dtype=np.float64)
r244_eye = torch.eye(192)
for start244 in range(0, 192, 12):
    cols244 = np.arange(start244, min(start244+12, 192))
    b244 = len(cols244)
    perturb244 = torch.cat([r244_eye[cols244], -r244_eye[cols244]], 0)
    perturb244 = perturb244.repeat_interleave(len(r244_sel), dim=0)
    pre244 = r244_pre.repeat(2*b244, 1, 1) + r244_eps*perturb244[:, None, :]
    ids244 = np.tile(r244_ids, 2*b244)
    pred244 = np242(r244_read_pre(pre244, ids244)).reshape(
        2, b244, len(r244_sel), 4
    )
    r244_J[:, :, cols244] = (
        (pred244[0]-pred244[1])/(2*r244_eps)
    ).transpose(1, 2, 0)

# Regress two native-unit response directions, penalising cross-effects.
r244_weights = np.array([1., 1., 5., 5.])
r244_X = (r244_J*r244_weights[None, :, None]).reshape(-1, 192)
r244_targets = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
r244_Y = np.tile(
    (r244_targets*r244_weights[:, None])[None], (len(r244_sel), 1, 1)
).reshape(-1, 2)
r244_gram = r244_X.T @ r244_X / len(r244_X)
r244_rhs = r244_X.T @ r244_Y / len(r244_X)
r244_dirs = {}
print("=== PRE-CODA READOUT ROUTING ===")
for ridge244 in [0.01, 0.1, 1.0]:
    reg244 = ridge244*np.trace(r244_gram)/192
    D244 = np.linalg.solve(r244_gram + reg244*np.eye(192), r244_rhs)
    r244_dirs[ridge244] = D244
    response244 = np.einsum("nkj,jd->nkd", r244_J, D244)
    print("ridge", ridge244,
          "direction norms", np.linalg.norm(D244, axis=0).round(4))
    print("mean responses [glucose direction, MAP direction]:")
    print(response244.mean(0).round(4))
    print("response SD:")
    print(response244.std(0).round(4))

print("\n=== ORIGINAL POSTERIOR TOKEN RETURN ===")
u244 = actions230[("insulin", 1.0)]
t244 = original_token231(u244)
print(type(t244).__name__,
      tuple(t244.shape) if hasattr(t244, "shape")
      else [(type(x).__name__, tuple(x.shape)) for x in t244])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 244: numerical readout routing\n"
            "- All-bundle constraints reduce rate-contrast slopes to ~0.29 insulin/~0.40 NE at penalty 1.\n"
            "- Factual feature RMS is ~0.215 insulin/~0.087 NE; amplification requires official validation.\n"
            "- Fitting pre-coda glucose/MAP directions by finite-difference response regression, without autograd.\n"
            "- No weights changed. Python runs: 244; forward passes: 17.\n")
```