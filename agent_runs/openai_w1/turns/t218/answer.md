The saved branch uses gated neurons to estimate dose changes from dose-and-context features, so it may provide a more selective signal than the ordinary value-head features. Before editing, I’ll measure its factual variation, check responses across dose multipliers, and see whether two latent coordinates can be replaced without disturbing the original projection. Small factual signals and consistent held-out dose responses would support a carrier experiment.

```python
import numpy as np
import torch
from scipy.special import ndtr
from pathlib import Path

def gelu195(x):
    return x * ndtr(x)

lc_branch = {}
for drug, spec in rp_branch_specs.items():
    def branch195(X, s=spec):
        X = np.asarray(X, dtype=float)
        g = X @ np.asarray(s["gw"]) + float(s["gb"])
        r = X @ np.asarray(s["rw"]) + float(s["rb"])
        return gelu195(g+r) - gelu195(g)
    q = branch195(rp_input)
    lc_branch[drug] = q
    active = np.asarray(rp_cf_masks[drug], dtype=bool)
    print("\nBRANCH", drug, "factual all RMS", float(np.sqrt(np.mean(q*q))),
          "active RMS", float(np.sqrt(np.mean(q[active]**2))),
          "inactive RMS", float(np.sqrt(np.mean(q[~active]**2))))
    print("factual quantiles", np.quantile(q, [.01, .1, .5, .9, .99]))
    for mult in (0., .5, 1.5, 2., 3.):
        qc = branch195(rp_cf_input[(drug, mult)])
        mask = active & np.asarray(rp_hold, dtype=bool)
        d = qc[mask] - q[mask]
        print(" multiplier", mult, "heldout contrast quantiles",
              np.round(np.quantile(d, [.1, .5, .9]), 4))

lc_Z = np.asarray(r168_zactual, dtype=float)
lc_train = np.asarray(rp_train, dtype=bool)
lc_hold = np.asarray(rp_hold, dtype=bool)
lc_dim_scores = []
for j in range(32):
    rest = [k for k in range(32) if k != j]
    X = np.column_stack([lc_Z[:, rest], np.ones(len(lc_Z))])
    A = X[lc_train].T @ X[lc_train] + 1e-5*np.eye(32)
    beta = np.linalg.solve(A, X[lc_train].T @ lc_Z[lc_train, j])
    err = X[lc_hold] @ beta - lc_Z[lc_hold, j]
    lc_dim_scores.append((float(np.sqrt(np.mean(err*err))), j))
lc_dim_scores.sort()
lc_dims = np.array([x[1] for x in lc_dim_scores[:2]])
lc_rest = np.array([j for j in range(32) if j not in lc_dims])
X = np.column_stack([lc_Z[:, lc_rest], np.ones(len(lc_Z))])
lc_comp = np.linalg.solve(
    X[lc_train].T @ X[lc_train] + 1e-5*np.eye(31),
    X[lc_train].T @ lc_Z[lc_train][:, lc_dims])
lc_zin0 = np.asarray(load_original("z_in.weight"), dtype=float)
lc_zbias0 = np.asarray(load_original("z_in.bias"), dtype=float)
lc_zin_base = lc_zin0.copy()
lc_zbias_base = lc_zbias0 + lc_zin0[:, lc_dims] @ lc_comp[-1]
lc_zin_base[:, lc_rest] += lc_zin0[:, lc_dims] @ lc_comp[:-1].T
lc_zin_base[:, lc_dims] = 0.
base_token = lc_Z @ lc_zin0.T + lc_zbias0
comp_token = lc_Z @ lc_zin_base.T + lc_zbias_base
print("\nCARRIER DIMS", lc_dims, "best one-coordinate errors", lc_dim_scores[:5])
print("joint compensation heldout token RMS error",
      float(np.sqrt(np.mean((comp_token[lc_hold]-base_token[lc_hold])**2))))

print("\nENCODER PARAMETERS")
print([(n, SHAPES[n]) for n in NAMES if n.startswith("encode.")])
print("atom helper constants", r95_correct_atom.__code__.co_consts[:12])
print("atom helper references", r95_correct_atom.__code__.co_names)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 195\nMeasured gated branch factual and held-out multiplier responses; selected two predictable posterior coordinates and fitted a closed-form z_in compensation from the remaining coordinates. No edits installed; compensation and carrier behavior must be validated before GPU testing.\n")
```
