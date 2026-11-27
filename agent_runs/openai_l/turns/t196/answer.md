Across all dose multipliers, the model’s own-drug responses are much smaller than the record estimates and flatten at higher doses. The records also support little insulin effect on MAP; norepinephrine’s cross-glucose estimates are less stable. I’ll fit candidate value-readout edits with strong factual anchoring and a weaker cross-drug constraint. Consistent dose slopes and held-out factual loss below the limit would support this approach.
```python
import numpy as np
from scipy.special import softmax as softmax179, logsumexp
from scipy.linalg import solve as solve179
from threadpoolctl import threadpool_limits

def array179(x):
    return np.asarray(x, dtype=np.float64)

H179 = array179(hidden_edited173)
L179 = array179(logits_edited173)
Lorig179 = array179(logits_original173)
C179 = array179(centres_train67)
B179 = np.asarray(observed_bins82, dtype=int)
events179 = measure67.event.to_numpy()
stays179 = measure67.stay_id.to_numpy()
rng179 = np.random.default_rng(179)
ids179 = rng179.permutation(np.unique(stays179))
fitmask179 = np.isin(stays179, ids179[:80])
valmask179 = ~fitmask179
X179 = np.concatenate([H179, np.ones((*H179.shape[:2], 1))], axis=2)
Yfact179 = Lorig179 - L179
Yfact179 -= Yfact179.mean(axis=2, keepdims=True)

def mean_tilt179(logits, centres, target):
    # Tilt only occupied-value bins, preserving the no-value probability.
    positive = logits[..., 1:]
    cc = centres[None, :, 1:]
    p0 = softmax179(logits, axis=-1)[..., 0]
    goal = target / np.maximum(1 - p0, 1e-12)
    goal = np.clip(goal, cc[..., 0] + 1e-5, cc[..., -1] - 1e-5)
    lo = np.full(goal.shape, -2.0)
    hi = np.full(goal.shape, 2.0)
    for _ in range(38):
        eta = (lo + hi) / 2
        pm = softmax179(positive + eta[..., None] * cc, axis=-1)
        mm = (pm * cc).sum(-1)
        lo = np.where(mm < goal, eta, lo)
        hi = np.where(mm >= goal, eta, hi)
    eta = (lo + hi) / 2
    delta = np.zeros_like(logits)
    normalizer = (logsumexp(positive + eta[..., None] * cc, axis=-1)
                  - logsumexp(positive, axis=-1))
    delta[..., 1:] = eta[..., None] * cc - normalizer[..., None]
    return delta - delta.mean(-1, keepdims=True)

XF179 = X179[:, fitmask179].reshape(-1, 193)
YF179 = Yfact179[:, fitmask179].reshape(-1, 17)
with threadpool_limits(limits=8):
    CF179 = XF179.T @ XF179 / len(XF179)
    RF179 = XF179.T @ YF179 / len(XF179)

CD179 = np.zeros((193, 193))
RD179 = np.zeros((193, 17))
mass179 = 0.0
targets179 = {}
for drug179 in ("insulin", "norepinephrine"):
    base179 = cf_features81[(drug179, 1.0)]
    rows179 = np.asarray(base179["rows"], dtype=int)
    own179 = np.isin(events179[rows179],
                    ("glucose_finger", "glucose_lab") if drug179 == "insulin"
                    else ("map_arterial", "map_cuff"))
    beta179 = np.where(own179, -1.95 if drug179 == "insulin" else 3.1, 0.0)
    rowweight179 = np.where(own179, 1.0, 0.25)
    bm179 = array179(base179["means"])
    bh179 = array179(base179["hidden"])
    for factor179 in (0.0, 0.5, 1.5, 2.0, 3.0):
        case179 = cf_features81[(drug179, factor179)]
        dr179 = np.asarray(case179["delta_rate"], dtype=float)
        wanted179 = bm179 + beta179[None, :] * dr179[None, :]
        target179 = mean_tilt179(array179(case179["logits"]), C179[rows179], wanted179)
        target179 -= Yfact179[:, rows179]
        targets179[(drug179, factor179)] = target179
        diff179 = array179(case179["hidden"]) - bh179
        diff179 = np.concatenate([diff179, np.zeros((*diff179.shape[:2], 1))], axis=2)
        use179 = fitmask179[rows179]
        xx179 = diff179[:, use179].reshape(-1, 193)
        yy179 = target179[:, use179].reshape(-1, 17)
        ww179 = np.tile(rowweight179[use179], H179.shape[0])
        ww179 *= 0.5 if factor179 == 3.0 else 1.0
        with threadpool_limits(limits=8):
            CD179 += xx179.T @ (xx179 * ww179[:, None])
            RD179 += xx179.T @ (yy179 * ww179[:, None])
        mass179 += ww179.sum()
CD179 /= mass179
RD179 /= mass179

def nll179(logits):
    pp = softmax179(logits, axis=-1).mean(0)
    return -np.log(np.maximum(pp[np.arange(len(B179)), B179], 1e-30))

def prediction179(case, correction):
    hh = array179(case["hidden"])
    xx = np.concatenate([hh, np.ones((*hh.shape[:2], 1))], axis=2)
    pp = softmax179(array179(case["logits"]) + xx @ correction, axis=-1)
    rr = np.asarray(case["rows"], dtype=int)
    return (pp * C179[rr][None]).sum(-1).mean(0)

def slopes179(correction, drug, factor, sensor, use_validation=True):
    base = cf_features81[(drug, 1.0)]
    case = cf_features81[(drug, factor)]
    rr = np.asarray(base["rows"], dtype=int)
    take = events179[rr] == sensor
    if use_validation:
        take &= valmask179[rr]
    dr = np.asarray(case["delta_rate"], dtype=float)[take]
    shift = (prediction179(case, correction) - prediction179(base, correction))[take]
    return float(np.dot(dr, shift) / max(np.dot(dr, dr), 1e-12))

original_nll179 = nll179(Lorig179)
print("Factual covariance eigenvalues:",
      np.linalg.eigvalsh(CF179)[:8].round(10).tolist())
print("Split:", int(fitmask179.sum()), "fit readings;",
      int(valmask179.sum()), "validation readings")
ridge179 = 1e-7 * np.trace(CF179[:192, :192]) / 192
penalty179 = np.eye(193) * ridge179
penalty179[-1, -1] = 0
readout179 = {}
for lam179 in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
    with threadpool_limits(limits=8):
        correction179 = solve179(CF179 + lam179 * CD179 + penalty179,
                                RF179 + lam179 * RD179, assume_a="pos")
    readout179[lam179] = correction179
    nl179 = nll179(L179 + X179 @ correction179)
    pair179 = {}
    for drug179, sensor179 in (("insulin", "glucose_finger"),
                              ("norepinephrine", "map_arterial")):
        pair179[drug179] = [round(slopes179(correction179, drug179, fac, sensor179), 3)
                           for fac in (0.0, 0.5, 1.5, 2.0, 3.0)]
    print("lambda", lam179, "weight_norm", round(float(np.linalg.norm(correction179)), 2),
          "NLL_delta fit/val", [
              round(float((nl179-original_nll179)[m].mean()), 6)
              for m in (fitmask179, valmask179)],
          "primary slopes [x0,x.5,x1.5,x2,x3]", pair179,
          "secondary x2", [
              round(slopes179(correction179, "insulin", 2.0, "glucose_lab"), 3),
              round(slopes179(correction179, "norepinephrine", 2.0, "map_cuff"), 3)])
print("No weights saved.")
```