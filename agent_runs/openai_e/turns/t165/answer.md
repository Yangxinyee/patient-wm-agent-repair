The fresh sensitivities reproduce the anchored decoder and differ only slightly from the archived sensitivities. I can now fit a closed-form correction that penalizes factual distribution changes and insulin-response changes while targeting the missing norepinephrine response. The held-out local results will show whether that trade-off is feasible; only GPU rollouts can establish whether it works over time.

```python
import numpy as np
import scipy.linalg as sla
from threadpoolctl import threadpool_limits
from pathlib import Path
import json, time

t144 = time.time()
K144 = Jm143.shape[1]
P144 = Binput142.shape[1]
N144 = K144 * P144
Fread144 = Fmean142[src141]
fit144 = fit_read141
events144 = head_meta["event"].to_numpy()
map144 = np.isin(events144, ["map_arterial", "map_cuff"])
sd144 = np.sqrt(np.maximum(
    (prob143 * centres141**2).sum(axis=1)
    - (prob143 * centres141).sum(axis=1)**2, 1.0
))

Xparts144, Jparts144, Yparts144, Wparts144 = [], [], [], []
group_info144 = []
for key144 in Dmean142:
    drug144, factor144 = key144
    present144 = cf_frame[
        "has_insulin" if drug144 == "insulin" else "has_norepinephrine"
    ].to_numpy(dtype=bool)
    direct144 = (cfhead141 >= 0)
    direct144 &= present144[np.maximum(cfhead141, 0)]
    groups144 = [("all", direct144, 4.0, 0.0)] if drug144 == "insulin" else [
        ("MAP", direct144 & map144, 1.0 if factor144 == 2.0 else 4.0,
         3.0 if factor144 == 2.0 else 1.6),
        ("glucose guard", direct144 & ~map144, 4.0, 0.0)
    ]
    for label144, mask144, weight144, beta144 in groups144:
        rows144 = np.flatnonzero(mask144 & fit144)
        if not len(rows144):
            continue
        cf144 = cfhead141[rows144]
        target144 = np.zeros(len(rows144), dtype=np.float64)
        if beta144:
            target144 = beta144 * arr141(dose_contrasts46[key144])[cf144, 1]
        Xparts144.append(Dmean142[key144][cf144])
        Jparts144.append(Jm143[rows144] / sd144[rows144, None])
        Yparts144.append(target144 / sd144[rows144])
        Wparts144.append(np.full(len(rows144), weight144 / len(rows144)))
        group_info144.append((key144, label144, len(rows144), beta144))

Xa144 = np.concatenate(Xparts144)
Ja144 = np.concatenate(Jparts144)
Ya144 = np.concatenate(Yparts144)
Wa144 = np.concatenate(Wparts144)
Xf144 = Fread144[fit144]
Sf144 = fisher143[fit144] / int(fit144.sum())
print("regression dimensions", N144, "factual rows", len(Xf144),
      "contrast rows", len(Xa144))
print("contrast groups", group_info144)

Hf144 = np.empty((N144, N144), dtype=np.float64)
Ha144 = np.empty_like(Hf144)
rhs144 = np.empty((K144, P144), dtype=np.float64)
with threadpool_limits(limits=8):
    for i144 in range(K144):
        si144 = slice(i144 * P144, (i144 + 1) * P144)
        rhs144[i144] = Xa144.T @ (Wa144 * Ja144[:, i144] * Ya144)
        for j144 in range(i144, K144):
            sj144 = slice(j144 * P144, (j144 + 1) * P144)
            hf144 = Xf144.T @ (Xf144 * Sf144[:, i144, j144, None])
            ha144 = Xa144.T @ (
                Xa144 * (Wa144 * Ja144[:, i144] * Ja144[:, j144])[:, None]
            )
            Hf144[si144, sj144] = hf144
            Ha144[si144, sj144] = ha144
            if i144 != j144:
                Hf144[sj144, si144] = hf144.T
                Ha144[sj144, si144] = ha144.T

    fits144, metrics144 = {}, {}
    for anchor144 in [0.3, 1.0, 3.0, 10.0, 30.0]:
        H144 = Ha144 + anchor144 * Hf144
        H144.flat[::N144 + 1] += 1e-8
        C144 = sla.solve(H144, rhs144.ravel(), assume_a="pos").reshape(K144, P144)
        fits144[anchor144] = C144
        ampl144 = Fread144 @ C144.T
        dlog144 = np.einsum("rbk,rk->rb", centred_J143, ampl144)
        dm144 = np.sum(Jm143 * ampl144, axis=1)
        held144 = ~fit144
        metric144 = {
            "norm": float(np.linalg.norm(C144)),
            "heldout_KL_linear": float(np.mean(
                .5 * np.sum(prob143[held144] * dlog144[held144]**2, axis=1)
            )),
            "heldout_mean_rms": {
                ev: float(np.sqrt(np.mean(dm144[held144 & (events144 == ev)]**2)))
                for ev in ["map_arterial", "map_cuff", "glucose_finger", "glucose_lab"]
                if np.any(held144 & (events144 == ev))
            }
        }
        for factor144 in [2.0, 0.5]:
            key144 = ("norepinephrine", factor144)
            use144 = (cfhead141 >= 0) & map144 & held144
            use144 &= ne_mask141[np.maximum(cfhead141, 0)]
            rows144 = np.flatnonzero(use144)
            cf144 = cfhead141[rows144]
            dd144 = arr141(dose_contrasts46[key144])[cf144, 1]
            response144 = np.sum(
                Jm143[rows144] * (Dmean142[key144][cf144] @ C144.T), axis=1
            )
            metric144[f"NE_{factor144}_slope"] = float(dd144 @ response144 / (dd144 @ dd144))
        metrics144[anchor144] = metric144

print("held-out local frontier")
for anchor144, metric144 in metrics144.items():
    print(anchor144, json.dumps(metric144))
print("elapsed seconds", round(time.time() - t144, 2))

with open("/work/notes.md", "a") as f144:
    f144.write(
        "\n## Run 144: closed-form multi-direction NE correction frontier\n"
        "Fit 14-output admission-null regressions with factual Fisher-distribution penalties "
        "and insulin/glucose guards, using stay-held-out local validation. "
        "Increment targets are NE beta 3.0 for doubling and 1.6 for halving, reflecting "
        "record-derived schedule effects minus the insulin-only baseline's existing responses. "
        f"Local frontier: {json.dumps(metrics144)}. "
        "No weights changed; local metrics are not rollout validation. "
        "Python runs: 144; forwards: 15.\n"
    )
```