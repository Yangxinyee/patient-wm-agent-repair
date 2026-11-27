The measured output sensitivities are small, so matching the estimated effects may require large weight changes. That raises saturation and factual-fit risks. I’ll compare several closed-form directions at different strengths before installing any response edit. If finite-dose responses remain weak despite increasing strength, I’ll reject this pathway rather than rely on its linear approximation.

```python
import numpy as np, torch

assert completed230 == 192
directions231 = {}
for drug in ["insulin", "norepinephrine"]:
    mask = drug_labels215[cal_idx230] == drug
    rates = rate_vector215[cal_idx230][mask]
    J = Jpair230[mask]
    native = native_pair230[mask]
    target = causal_targets215[drug][None, :] * rates[:, None] - native
    avgJ = J.sum(0) / rates.sum()
    avgY = target.sum(0) / rates.sum()
    weights = np.array([1., .4, 2., .8]) if drug == "insulin" else np.array([.8, .4, 1., .4])

    X = avgJ * weights[:, None]
    y = avgY * weights
    vmean = X.T @ np.linalg.solve(X @ X.T + 1e-6 * np.eye(4), y)

    Xr = (J * weights[None, :, None]).reshape(-1, 192)
    yr = (target * weights[None, :]).reshape(-1)
    vrow = np.linalg.solve(Xr.T @ Xr / len(rates) + 1e-5 * np.eye(192),
                          Xr.T @ yr / len(rates))

    primary = [0, 2, 3] if drug == "insulin" else [2, 0, 1]
    Xp, yp = avgJ[primary], avgY[primary]
    vprimary = Xp.T @ np.linalg.solve(Xp @ Xp.T + 1e-6 * np.eye(3), yp)
    for method, v in [("mean", vmean), ("row", vrow), ("primary", vprimary)]:
        directions231[(method, drug)] = v
        print(method, drug, "norm", round(float(np.linalg.norm(v)), 3),
              "linear predicted slopes",
              np.round(avgJ @ v + native.sum(0) / rates.sum(), 4))

def response_candidate231(method, gain=1.):
    result = dict(route228)
    ow = np.array(route228["core.prelude.0.attn.out.weight"], copy=True)
    for di, drug in enumerate(["insulin", "norepinephrine"]):
        ow[:, response_cols228[di]] = gain * directions231[(method, drug)]
    result["core.prelude.0.attn.out.weight"] = ow
    return result

finite231 = {}
for method in ["mean", "row", "primary"]:
    for gain in [.25, 1., 4.]:
        candidate = response_candidate231(method, gain)
        hf, of = replay_route228(candidate, Ubase215)
        hd, od = replay_route228(candidate, Udouble230)
        mf, md = means229(hf, of), means229(hd, od)
        finite231[(method, gain)] = (mf, md)
        print("\nFINITE", method, gain, "factual mean RMS",
              np.round(np.sqrt(np.mean((mf - route_means229)**2, 0)), 4))
        for drug in ["insulin", "norepinephrine"]:
            mask = drug_labels215 == drug
            print(drug, "x2 slopes",
                  np.round((md - mf)[mask].sum(0) / rate_vector215[mask].sum(), 4))

np.savez("/work/response_directions231.npz",
         **{method + "_" + drug: v for (method, drug), v in directions231.items()})
with open("/work/notes.md", "a") as f231:
    f231.write("\n## Python 231 — finite-strength direct-column checks\n"
               "- Measured sensitivity is weak; calibrated directions therefore have potentially disruptive norms.\n"
               "- Fitted mean, rowwise ridge, and primary-sensor directions using closed-form regressions only.\n"
               "- Tested finite gains .25, 1 and 4 on all retained action contexts before installation.\n"
               "- No response candidate installed; local responses and factual mean RMS are not GPU certification.\n")
```