The original rollouts are substantially too weak: insulin doubling predicts about −3.02 versus the provisional record-based target of −10.68, and norepinephrine doubling about +0.43 versus +4.33. I’ll test whether small latent-state changes can selectively affect glucose or MAP. Stable, sensor-specific responses would support this route; strong cross-effects or nonlinearity would argue against it.

```python
import time
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(8)

print("=== DOSE UNITS ===")
for drug88 in ["insulin", "norepinephrine"]:
    v88 = np.asarray(rate_state81[drug88])
    positive88 = v88[v88 > 0]
    print(drug88, "state-rate quantiles:",
          np.quantile(positive88, [0, .25, .5, .75, 1]))
    for factor88 in [2.0, .5]:
        c88 = cf_features81[(drug88, factor88)]
        print("factor", factor88, "delta-rate quantiles:",
              np.quantile(c88["delta_rate"], [0, .5, 1]))

rng88 = np.random.default_rng(8801)
rows88 = []
for event88 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    eligible88 = np.flatnonzero(
        (events74 == event88)
        & ((rate_state81["insulin"][pi73] > 0)
           | (rate_state81["norepinephrine"][pi73] > 0))
    )
    rows88.extend(rng88.choice(eligible88, min(25, len(eligible88)), replace=False))
rows88 = np.sort(np.asarray(rows88, dtype=np.int64))
ids88 = np.unique(pi73[rows88])
positions88 = np.searchsorted(ids88, pi73[rows88])
ctx88, hf88, mem88, ages88, mask88 = bundle_inputs74(ids88)
post88 = mlp63(torch.cat([ctx88, e_valid60[ids88], u_valid60[ids88]], -1),
               "z_post")[:, :32]

def latent_probe88(addition):
    z88 = post88 + torch.as_tensor(addition, dtype=torch.float32)
    out88, h88 = transition64(
        hf88, e_valid60[ids88], u_valid60[ids88], z88,
        mem88, ages88, mask88
    )
    readctx88, _ = context64(
        out88[positions88], h88[positions88], dtread73[rows88]
    )
    means88, _, _, _ = decode67(
        readctx88, const_train67[rows88], trans_train67[rows88],
        centres_train67[rows88], epsread73[:, rows88],
        return_features=True
    )
    return means88.mean(0).numpy()

start88 = time.time()
base88 = latent_probe88(np.zeros(32, dtype=np.float32))
G88 = np.empty((len(rows88), 32), dtype=np.float64)
for j88 in range(32):
    perturb88 = np.zeros(32, dtype=np.float32)
    perturb88[j88] = .10
    G88[:, j88] = (latent_probe88(perturb88) - latent_probe88(-perturb88)) / .20

scales88 = np.where(np.isin(events74[rows88],
                            ["glucose_finger", "glucose_lab"]), 30., 10.)
Gscaled88 = G88 / scales88[:, None]
latent_dirs88 = {}
print("\n=== LATENT SELECTIVITY ===")
for drug88 in ["insulin", "norepinephrine"]:
    relevant88 = np.isin(
        events74[rows88],
        ["glucose_finger", "glucose_lab"] if drug88 == "insulin"
        else ["map_arterial", "map_cuff"]
    )
    target88 = relevant88.astype(float) / scales88
    gram88 = Gscaled88.T @ Gscaled88
    reg88 = 1e-4 * np.trace(gram88) / 32
    direction88 = np.linalg.solve(
        gram88 + reg88 * np.eye(32), Gscaled88.T @ target88
    ).astype(np.float32)
    latent_dirs88[drug88] = direction88
    predicted88 = G88 @ direction88
    nonlinear88 = latent_probe88(direction88) - base88
    print(drug88, "direction norm", round(float(np.linalg.norm(direction88)), 4))
    for event88 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        m88 = events74[rows88] == event88
        print(event88,
              "linear mean/std", np.round([predicted88[m88].mean(),
                                          predicted88[m88].std()], 4),
              "finite-change mean/std", np.round([nonlinear88[m88].mean(),
                                                 nonlinear88[m88].std()], 4))
print("probe seconds:", round(time.time() - start88, 1))
print("base reconstruction RMSE:",
      float(np.sqrt(np.mean((base88 - means_fact80[:, rows88].mean(0).numpy())**2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 88 / latent-direction finite-difference probe\n"
            "Original tuning rollout summaries: I2=-3.0206, Ihalf=1.6282 "
            "(60 reached stays); NE2=.4330, NEhalf=-.8431 (26 reached stays). "
            "Record-kernel targets remain provisional I2~-10.675/NE2~4.326. "
            "Probing sensor-selective latent directions with no gradients and no edits.\n")
```