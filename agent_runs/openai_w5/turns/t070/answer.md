That run stopped before testing anything because the context helper returns a tuple. I’ll use the already assembled context tensor and rerun the held-out decoding test. The hypothesis remains untested: accurate decoding at drug bundles would support a targeted posterior edit, while poor decoding would rule it out.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np, pandas as pd, torch

torch.set_num_threads(8)
rows55 = np.asarray(rows4, dtype=int)
ctx55 = ctx4
PH55 = as_np(posterior_features4(U4))
z55 = wm_lin(torch.from_numpy(PH55), "z_post.2")[:, :32].numpy()
print("Posterior mean reconstruction RMSE:",
      float(np.sqrt(np.mean((z55 - s4["x"][rows55, 960:992])**2))))

hour55 = np.floor(s4["hour"][rows55] + 1e-5).astype(int)
idx55 = pd.MultiIndex.from_arrays(
    [s4["stay_id"][rows55], hour55], names=X.index.names)
valid55 = idx55.isin(X.index) & (hour55 >= 0) & (hour55 < 24)
rate55 = np.asarray(dose_actual4[rows55], dtype=float).copy()
expected55 = np.zeros_like(rate55)
for d, drug in enumerate(["insulin", "norepinephrine"]):
    expected55[valid55, d] = nuisance_models[drug].predict(
        X.loc[idx55[valid55]])
present55 = np.zeros((len(s4["stay_id"]), 2), dtype=bool)
for d, drug in enumerate(["insulin", "norepinephrine"]):
    jj = acttable4.loc[acttable4.event == drug, "j"].to_numpy(dtype=int)
    present55[jj, d] = True
present55 = present55[rows55]
innovation55 = np.where(present55, rate55 - expected55, 0.0)
ids55 = np.unique(s4["stay_id"])
train55 = valid55 & np.isin(s4["stay_id"][rows55], ids55[:70])
test55 = valid55 & np.isin(s4["stay_id"][rows55], ids55[70:])
scale55 = np.array([norm_meta[(d, "rate")]["iqr"]
                    for d in ["insulin", "norepinephrine"]])
feature55 = {
    "posterior_hidden": PH55,
    "action_u": U4[rows55].numpy(),
    "ctx_plus_u": np.concatenate([ctx55.numpy(), U4[rows55].numpy()], axis=1),
    "post_state_out_z": np.concatenate(
        [s4["x"][rows55, :192], s4["x"][rows55, 960:992]], axis=1),
}
probe_fits55 = {}
for target_name, target in [("rate", rate55), ("innovation", innovation55)]:
    yy = target / scale55
    for feature_name, ff in feature55.items():
        trials55 = []
        for aa in [0.01, 1.0, 100.0]:
            reg = Ridge(alpha=aa).fit(ff[train55], yy[train55])
            pred55 = reg.predict(ff[test55])
            per_drug = []
            for d in range(2):
                active = present55[test55, d]
                r2 = r2_score(yy[test55, d][active], pred55[:, d][active])
                rmse = np.sqrt(np.mean(
                    (yy[test55, d][active] - pred55[:, d][active])**2))
                per_drug.append((round(float(r2), 4), round(float(rmse), 4),
                                 int(active.sum())))
            trials55.append((aa, per_drug))
            probe_fits55[(target_name, feature_name, aa)] = reg
        print(target_name, feature_name, trials55)

print("Rate scale:", scale55)
print("Active innovation SD:",
      [float(innovation55[valid55 & present55[:, d], d].std())
       for d in range(2)])
print("Drug-bundle counts, train/test:",
      int((train55 & present55.any(1)).sum()),
      int((test55 & present55.any(1)).sum()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart, runs 55–59\n"
            "- Re-read architecture and inspected local recurrence.\n"
            "- Run 58 stopped on context tuple/tensor mismatch; no edit.\n"
            "- Run 59 tests dose/innovation decoding at drug bundles "
            "on stay-held-out splits.\n")
```
