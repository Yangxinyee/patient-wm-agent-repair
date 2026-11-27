The atom encoding now matches exactly, and the GPU-anchored NE doubling features reproduce the earlier contrasts exactly. I’ll fit a nonlinear detector with both dose changes balanced, insulin contrasts guarded, and whole-stay cross-validation. It is promising only if factual leakage falls materially without losing either dose response.

```python
from pathlib import Path
import numpy as np, torch, pandas as pd
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

torch.set_num_threads(8)
hidden_cf244 = {("norepinephrine", 2.0): hidden_ne2_243}
for key, data in cf_train192.items():
    if key in hidden_cf244:
        continue
    ids = np.asarray(data["ids"], dtype=int)
    ee, uu = encoded_bundles243(
        ids, drug=key[0], factor=key[1], **rule243
    )
    zf = posterior94(ctx_base191[ids], obs191[ids], act191[ids])
    zz = z_base191[ids] + posterior94(ctx_base191[ids], ee, uu) - zf
    hidden, raw = prelude_hidden93(h_base191[ids], ee, uu, zz)
    check = np243(core_norm(torch.from_numpy(raw), "core.prelude_norm"))
    error = float(np.max(np.abs(check-np243(data["E"]))))
    print("contrast alignment", key, error)
    assert error < 1e-4
    hidden_cf244[key] = hidden

flat244 = hidden_base242.reshape(-1, 768).astype(np.float64)
scale244 = np.maximum(np.sqrt(np.mean(flat244**2, axis=0)), 1e-3)
X244 = np.column_stack([flat244/scale244, np.ones(len(flat244))])
initial_mask244 = np.repeat(np.asarray(initial_admission198, bool), 4)
_, sv244, vt244 = np.linalg.svd(
    X244[initial_mask244], full_matrices=True
)
rank244 = int(np.sum(sv244 > sv244[0]*1e-6))
B244 = vt244[rank244:].T
F244 = X244 @ B244
fold244 = np.repeat(np.asarray(foldF197), 4)
print("initial guard rank", rank244, "remaining", B244.shape[1])

D244, Y244, folds_cf244 = {}, {}, {}
for key, data in cf_train192.items():
    ids = np.asarray(data["ids"], int)
    delta = hidden_cf244[key] - hidden_base242[ids]
    raw_design = np.column_stack([
        delta.reshape(-1, 768)/scale244,
        np.zeros(len(ids)*4)
    ])
    D244[key] = raw_design @ B244
    Y244[key] = np.repeat(np.asarray(data["dose_change"], float), 4)
    folds_cf244[key] = np.repeat(np.asarray(foldF197)[ids], 4)

nekeys244 = [("norepinephrine", 2.0), ("norepinephrine", .5)]
ikeys244 = [("insulin", 2.0), ("insulin", .5)]
dose2_244 = float(np.mean(Y244[nekeys244[0]]**2))
choices244 = [(fw, ridge) for fw in (1., 10., 100.)
              for ridge in (.001, .01, .1)]
records244 = []
fits244 = {}

with threadpool_limits(limits=8):
    for fold in range(5):
        fit = fold244 != fold
        hold = ~fit
        gf = F244[fit].T @ F244[fit] / (fit.sum()*dose2_244)
        gc = np.zeros_like(gf)
        rc = np.zeros(F244.shape[1])
        gi = np.zeros_like(gf)
        for key in nekeys244:
            m = folds_cf244[key] != fold
            x, y = D244[key][m], Y244[key][m]
            denominator = len(y)*float(np.mean(y*y))
            gc += x.T @ x / denominator
            rc += x.T @ y / denominator
        for key in ikeys244:
            m = folds_cf244[key] != fold
            x = D244[key][m]
            gi += x.T @ x / (len(x)*dose2_244)
        for fw, ridge in choices244:
            g = gc + fw*gf + 10.*gi
            g.flat[::len(g)+1] += ridge
            coef = solve(g, rc, assume_a="pos", check_finite=False)
            row = dict(fold=fold, factual_weight=fw, ridge=ridge)
            p = F244[hold] @ coef
            row["factual_RMS"] = float(np.sqrt(np.mean(p*p)))
            for key in nekeys244 + ikeys244:
                m = folds_cf244[key] == fold
                p, y = D244[key][m] @ coef, Y244[key][m]
                label = ("NE" if key[0] == "norepinephrine" else "I") + (
                    "2" if key[1] == 2. else "half"
                )
                row[label+"_slope"] = float(p @ y / (y @ y))
                row[label+"_RMS"] = float(np.sqrt(np.mean(p*p)))
            records244.append(row)

    # Fit each candidate on all training states after cross-validation.
    gf = F244.T @ F244 / (len(F244)*dose2_244)
    gc, gi = np.zeros_like(gf), np.zeros_like(gf)
    rc = np.zeros(F244.shape[1])
    for key in nekeys244:
        x, y = D244[key], Y244[key]
        denominator = len(y)*float(np.mean(y*y))
        gc += x.T @ x / denominator
        rc += x.T @ y / denominator
    for key in ikeys244:
        x = D244[key]
        gi += x.T @ x / (len(x)*dose2_244)
    for fw, ridge in choices244:
        g = gc + fw*gf + 10.*gi
        g.flat[::len(g)+1] += ridge
        coef = solve(g, rc, assume_a="pos", check_finite=False)
        affine = B244 @ coef
        fits244[(fw, ridge)] = np.r_[affine[:768]/scale244, affine[768]]

metrics244 = pd.DataFrame(records244)
summary244 = metrics244.groupby(["factual_weight", "ridge"]).mean(
    numeric_only=True
).drop(columns="fold").reset_index()
summary244["unit_factual"] = (
    summary244.factual_RMS / summary244.NE2_slope
)
summary244["half_ratio"] = (
    summary244.NEhalf_slope / summary244.NE2_slope
)
print("\nWHOLE-STAY CROSS-VALIDATION")
print(summary244.to_string(index=False))
print("CURRENT EDITS", edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 244: balanced nonlinear NE detector cross-validation\n"
        "Factual atom conventions and GPU-anchored dose contrasts verified "
        "exactly. Fitted 768-feature affine detectors by closed-form ridge, "
        "protecting only pure initial admissions and guarding both insulin "
        "contrasts. Five-fold whole-training-stay diagnostics:\n"
        + summary244.to_string(index=False)
        + "\nNo weights changed. Python runs: 244; forwards: 29.\n"
    )
```