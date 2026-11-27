The centered posterior edits barely improve dose responses, even when made much larger. This suggests the state-update path is attenuating or saturating them. I’ll test whether a dose-innovation readout in the more sensitive prior head can recover the missing response, first without changing the state and then with a small carrier.

```python
import time
import numpy as np
import torch
import torch.nn.functional as F

t74 = time.time()
old_mlp74 = mlp0
capture74 = []
def traced_mlp74(x, prefix, kind="gelu"):
    if prefix == "z_prior":
        capture74.append(x.detach().clone())
    return old_mlp74(x, prefix, kind)
mlp0 = traced_mlp74

def prior_features74(indices, u=None):
    capture74.clear()
    o, h = r59_core(indices, u=u)
    r59_features(o, h, indices)
    if not capture74:
        raise RuntimeError("Prior context was not captured.")
    ctx = capture74[-1]
    hidden = F.gelu(F.linear(ctx, W0["z_prior.0.weight"], W0["z_prior.0.bias"]))
    hidden = arr71(hidden)
    return np.c_[hidden, np.ones(len(hidden))]

all74 = np.arange(len(rate66))
stay74 = np.asarray(rr33_stays, dtype=np.int64)
train74 = stay74 % 5 != 0
dose_units74 = np.array([1.0, 0.1])
carrier74 = np.asarray(D56_candidates[0.001]).copy()
carrier74 /= np.linalg.norm(carrier74, axis=0, keepdims=True)
carrier74 *= np.array([0.05, 0.5])[None, :]
data74, decoders74, diagnostics74 = {}, {}, []

try:
    for mode, strength in [("original", None), ("small_carrier", 1.0),
                           ("larger_carrier", 4.0)]:
        if strength is None:
            W0.update({k: v.clone() for k, v in backup71.items()})
        else:
            set_post71(carrier74, strength)
        Z = prior_features74(all74)
        differences, changes, cf_indices, labels = [], [], [], []
        for d in [0, 1]:
            ii = drug_idx71[d]
            for multiplier in [0.0, 0.5, 2.0, 3.0]:
                cf = prior_features74(ii, cf_u71[d, multiplier])
                differences.append(cf - Z[ii])
                delta = np.zeros((len(ii), 2))
                delta[:, d] = rate66[ii, d] * (multiplier - 1.0)
                changes.append(delta)
                cf_indices.extend(ii.tolist())
                labels.extend([(d, multiplier)] * len(ii))
        DX, DY = np.vstack(differences), np.vstack(changes)
        cfi = np.asarray(cf_indices)
        ctr = train74[cfi]
        factual_target = np.asarray(rtrue66) / dose_units74
        contrast_target = DY / dose_units74
        data74[mode] = (Z, DX, DY, cfi, labels)
        z, dx = Z[train74], DX[ctr]
        qf, qc = z.T @ z, dx.T @ dx
        bf = z.T @ factual_target[train74]
        bc = dx.T @ contrast_target[ctr]
        balance = len(z) / max(len(dx), 1)
        for penalty in [0.01, 0.1, 1.0, 10.0]:
            gram = qf + penalty * balance * qc + 1e-4 * np.eye(Z.shape[1])
            coef = np.linalg.solve(gram, bf + penalty * balance * bc)
            coef *= dose_units74[None, :]
            decoders74[mode, penalty] = coef
            pred = Z @ coef
            cf_change = DX @ coef
            rmse, absent = [], []
            for d in [0, 1]:
                active = (~train74) & (rate66[:, d] > 0)
                inactive = (~train74) & (rate66[:, d] == 0)
                rmse.append(float(np.sqrt(np.mean(
                    (pred[active, d] - rtrue66[active, d])**2))))
                absent.append(float(np.sqrt(np.mean(pred[inactive, d]**2))))
            slopes = {}
            for d in [0, 1]:
                for mult in [0.0, 0.5, 2.0, 3.0]:
                    choose = np.array([lab == (d, mult) for lab in labels]) & (~ctr)
                    a, b = DY[choose, d], cf_change[choose, d]
                    slopes[d, mult] = round(float(a @ b / (a @ a + 1e-12)), 3)
            result = {"mode": mode, "penalty": penalty,
                      "active_RMSE": np.round(rmse, 4).tolist(),
                      "absent_RMS": np.round(absent, 4).tolist(),
                      "held_contrast_slopes": slopes}
            diagnostics74.append(result)
            print(result)
finally:
    mlp0 = old_mlp74
    W0.update(backup71)

print("Seconds:", round(time.time()-t74, 2), "Submitted edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 74: prior-feature dose decoding\n"
            "- Direct posterior corrections saturate and do not repair local slopes; none submitted.\n"
            "- Fitted preservation/contrast-constrained dose-innovation readouts in exact prior hidden features.\n"
            "- Compared original state path and two small explicit carriers; held-out diagnostics74 retained.\n"
            "- Python runs: 74; forward passes: 4; no submitted edits.\n")
```