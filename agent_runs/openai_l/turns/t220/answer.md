The dose contrasts use the correct record units. The apparent huge shifts came from interpreting logit corrections as complete logits, so the targets do not need rescaling. I’ll calibrate the dose directions in physical units and test their factual likelihood impact. Improvement, rather than merely unchanged predictions, would justify a stronger correction.
```python
from scipy.special import softmax, logsumexp

print("=== READING TABLE CANDIDATES ===")
for n in sorted(list(globals())):
    v = globals()[n]
    if hasattr(v, "columns") and len(v) == 4828:
        print(n, list(v.columns), v.head(2).to_dict("records"))

P199 = softmax(L179, axis=-1)
mean199 = np.mean(np.sum(P199 * C179[None], axis=-1), axis=0)
var199 = np.mean(np.sum(P199 * (C179[None] - mean199[None,:,None])**2,
                       axis=-1), axis=0)
physical199 = {}
gatefact199 = gatepred192
base_nll199 = nll179(L179)
obscentre199 = C179[np.arange(len(B179)), B179]

print("\n=== PHYSICAL CALIBRATION AND FACTUAL TEST ===")
for drug, sensor, beta in [("insulin","glucose_finger",-1.95),
                           ("norepinephrine","map_arterial",3.1)]:
    family = np.isin(events179,
        ["glucose_finger","glucose_lab"] if drug == "insulin"
        else ["map_arterial","map_cuff"])
    c1 = cf_features81[(drug, 1.0)]
    c2 = cf_features81[(drug, 2.0)]
    rr = np.asarray(c2["rows"], dtype=int)
    dr = np.asarray(c2["delta_rate"], dtype=float)
    take = (events179[rr] == sensor) & ~fitmask179[rr]
    X1 = np.concatenate([arr192(c1["input"]),
                        np.ones((*c1["input"].shape[:2],1))], axis=-1)
    X2 = np.concatenate([arr192(c2["input"]),
                        np.ones((*c2["input"].shape[:2],1))], axis=-1)
    original_shift = (arr192(c2["means"]) - arr192(c1["means"])).mean(axis=0)
    original_slope = float(np.dot(original_shift[take], dr[take]) /
                           np.dot(dr[take], dr[take]))
    print(drug, "original one-step slope", round(original_slope,4),
          "variance quantiles", np.quantile(var199[family],[.1,.5,.9]).round(3).tolist())
    for lam in [.03,.1,.3,1.0,3.0]:
        w = directions193[(drug,lam)].copy()
        dx = ((X2-X1) @ w).mean(axis=0)
        gain = float(np.dot(dx[take],dr[take])/np.dot(dr[take],dr[take]))
        w /= gain
        physical199[(drug,lam)] = w
        residual = (XF192 @ w)
        residual[:,~family] = 0
        rv = residual.mean(axis=0)
        hold = family & ~fitmask179
        err = obscentre199 - mean199
        fitted = float(np.dot(rv[hold],err[hold])/
                       max(np.dot(rv[hold],rv[hold]),1e-12))
        changes = {}
        for strength in [0,.5,1.0]:
            shift = strength * (beta-original_slope) * residual
            # Local diagnostic only: variance-adaptive tilt, not yet a weight edit.
            tilt = shift[...,None] * (C179[None]-mean199[None,:,None]) / np.maximum(var199[None,:,None],1)
            dn = nll179(L179 + tilt)-base_nll199
            changes[strength] = round(float(dn[~fitmask179].mean()),6)
        print(" lambda", lam, "raw-to-physical gain", round(gain,3),
              "held-out relevant residual SD", round(float(rv[hold].std()),4),
              "residual-error slope", round(fitted,3),
              "diagnostic total NLL deltas", changes)

print("All model weights unchanged.")
```
