Finger and lab surprises share one z-direction (cosine 1.00), and arterial and cuff share another (cosine 1.00). Only the gain differs: z moves 0.023 per mg/dL of finger surprise and 0.069 per mmHg of arterial surprise. The glucose and MAP directions are distinct (cosine about −0.3).

So z holds a latent glucose level and a latent MAP level, each shared across measurement sources. Injecting along these encoding directions should move both readouts of the same quantity, finger and lab (or arterial and cuff), which the evaluation also scores.

Before building the edit, the gates need to be safe:
1. Refit the NE-presence gate on a realistic dense grid: insulin rates in 0.25 steps up to 11.5 crossed with NE rates in 0.01 steps up to 3, plus singles and the empty bundle. I check the margin, and the margin when NE rates are limited to ≤2.
2. Check how large the rate readouts get on bundles where that drug is absent. Those values determine how negative the gate must push the units to keep them off.

```python
rNd = np.unique(np.r_[0.0, np.arange(0.01, 3.0001, 0.01)])
grid_pairs = [(a, None) for a in rI0] + [(None, b) for b in rNd] + [(a, b) for a in rI0[::2] for b in rNd[::3]] + [(None, None)]
def Up(a, b):
    at = ([ins_atom(float(a))] if a is not None else []) + ([ne_atom(float(b))] if b is not None else [])
    return U_of(at)
UG2 = np.stack([Up(a, b) for a, b in grid_pairs]); aG2 = np.array([np.nan if a is None else a for a, _ in grid_pairs]); bG2 = np.array([np.nan if b is None else b for _, b in grid_pairs])
yN2 = ~np.isnan(bG2); yI2 = ~np.isnan(aG2)
for C_ in [10.0, 1000.0]:
    for lim in [3.0, 2.0]:
        keep = np.isnan(bG2) | (bG2 <= lim)
        clf = LinearSVC(C=C_, max_iter=500000, tol=1e-8).fit(UG2[keep], yN2[keep]); sc = UG2[keep] @ clf.coef_.ravel() + clf.intercept_[0]
        print('NE gate C=%g lim %.1f: min pos %.3f max neg %.3f' % (C_, lim, sc[yN2[keep]].min(), sc[~yN2[keep]].max()))
scI2 = UG2 @ wgI + bgI; print('INS gate on dense grid: min pos %.3f max neg %.3f' % (scI2[yI2].min(), scI2[~yI2].max()))
fI_off = fI.predict(UG2[~yI2]); fN_off = fN.predict(UG2[~yN2])
print('fI on no-insulin bundles: range %.1f..%.1f ; fN on no-NE bundles: range %.2f..%.2f' % (fI_off.min(), fI_off.max(), fN_off.min(), fN_off.max()))
```