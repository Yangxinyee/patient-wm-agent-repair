The discrepancy is localized: drug bundles reconstruct the exposed posterior to about 0.0022 RMS, while other bundles account for the large error. The cached weights also match the originals.

That supports fitting a drug-gated correction, but factual fit still needs GPU validation. I’ll now fit dose decoders across several multipliers and test pre-action dose predictors on held-out training stays. Reliable decoding and small held-out residuals would support a centred edit.

```python
import numpy as np
import torch
from sklearn.linear_model import Ridge
from time import perf_counter

torch.set_num_threads(8)
t168 = perf_counter()
presence168 = (rates152 > 0).astype(np.float32)
active168 = np.flatnonzero(presence168.any(1))
unique_stays168 = np.unique(stays156)
train_stays168 = unique_stays168[::2]
fit_mask168 = np.isin(stays156, train_stays168)
hold_mask168 = ~fit_mask168

def encode_actions168(scales):
    rr = []
    for row in act_rows152:
        r = dict(row)
        j = 0 if r["event"] == "insulin" else 1
        r["rate"] = float(r["rate"]) * scales[j]
        r["dose"] = float(r["dose"]) * scales[j]
        rr.append(r)
    return encode_rows152(
        rr, act_groups152, [False] * len(rr), "encode_action",
        units=True, ru=(2, 3), zero_empty=True
    )

Uref168 = encode_actions168((1.0, 1.0))
print("Reference encoder RMS error:",
      float(torch.sqrt(torch.mean((Uref168-U166)**2))))
assert float(torch.sqrt(torch.mean((Uref168-U166)**2))) < 1e-5

scales168 = [(1., 1.)]
scales168 += [(s, 1.) for s in (0., .5, 1.5, 2., 3.)]
scales168 += [(1., s) for s in (0., .5, 1.5, 2., 3.)]
Uaug168 = {(1., 1.): Uref168.numpy()}
for sc in scales168[1:]:
    Uaug168[sc] = encode_actions168(sc).numpy()

empty168 = np.flatnonzero(~presence168.any(1))[::12]
decode_indices168 = np.concatenate((active168, empty168))
Xd168 = np.concatenate([Uaug168[sc][decode_indices168] for sc in scales168])
Yd168 = np.concatenate([
    rates152[decode_indices168] * np.asarray(sc, dtype=np.float32)
    for sc in scales168
])
Yp168 = np.tile(presence168[decode_indices168], (len(scales168), 1))
Md168 = np.tile(fit_mask168[decode_indices168], len(scales168))

dose_decoder168 = Ridge(alpha=.001).fit(Xd168[Md168], Yd168[Md168])
presence_decoder168 = Ridge(alpha=.001).fit(Xd168[Md168], Yp168[Md168])
print("\n=== HELD-STAY ACTION DECODING ===")
for label, reg, target in (
    ("rate", dose_decoder168, Yd168),
    ("presence", presence_decoder168, Yp168),
):
    err = reg.predict(Xd168[~Md168]) - target[~Md168]
    print(label, "RMSE per drug:", np.sqrt(np.mean(err**2, axis=0)),
          "99% absolute error:", np.quantile(np.abs(err), .99, axis=0))

policy_inputs168 = {
    "context": context.numpy(),
    "context+e": np.concatenate((context.numpy(), E166.numpy()), axis=1),
}
policy_results168 = []
print("\n=== HELD-STAY PRE-ACTION POLICY PREDICTION ===")
for j, drug in enumerate(("insulin", "norepinephrine")):
    present = presence168[:, j] > 0
    tr = present & fit_mask168
    te = present & hold_mask168
    for feature_name, xx in policy_inputs168.items():
        for alpha in (.01, .1, 1., 10., 100.):
            reg = Ridge(alpha=alpha).fit(xx[tr], rates152[tr, j])
            residual = rates152[te, j] - reg.predict(xx[te])
            rmse = float(np.sqrt(np.mean(residual**2)))
            policy_results168.append((rmse, j, feature_name, alpha, reg))
    best = min(r for r in policy_results168 if r[1] == j)
    print(drug, "best RMSE / features / alpha:",
          best[0], best[2], best[3],
          "held rate SD:", float(np.std(rates152[te, j])))
    print("top alternatives:",
          [(round(r[0], 4), r[2], r[3])
           for r in sorted([r for r in policy_results168 if r[1] == j],
                           key=lambda r: r[0])[:4]])

dose_decoder168.fit(Xd168, Yd168)
presence_decoder168.fit(Xd168, Yp168)
policy168 = {}
for j, drug in enumerate(("insulin", "norepinephrine")):
    best = min((r for r in policy_results168 if r[1] == j), key=lambda r: r[0])
    xx = policy_inputs168[best[2]]
    present = presence168[:, j] > 0
    policy168[drug] = (
        best[2], Ridge(alpha=best[3]).fit(xx[present], rates152[present, j])
    )
print("\nElapsed seconds:", round(perf_counter()-t168, 2))
```
