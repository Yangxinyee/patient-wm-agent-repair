Units 114, 0 and 10 are effectively dead: pre-activation stays below −2.3 on every row, with mean output ≤ 0.001 and zero measurable cost. So I'll repurpose unit 114 for insulin and unit 0 for NE.

Next is a better decoder for drug I. I'll use a minimum-variance matched filter, v ∝ Σ⁻¹μ:
- μ is the mean head-input response to the ×2 edit on reached glucose rows, per unit change in lag-1 dose;
- Σ is the within-cell factual covariance on glucose rows.

For a few regularisation levels I'll report how the decoder output responds to the dose changes in both insulin edits, and its factual noise. The decoder is good if the lag-1 response is about 1 in both edits, the noise sd is under about 1 dose unit, and leakage from the NE edit is small.

```python
def mvdr_dec(drug, codes, lam):
    dct = EDR[(drug, 2.0)]; gi_ = dct['gi']; rc_ = dct['reached'].astype(bool) & np.isin(cf[gi_], codes)
    mu_ = dct['dX'][rc_].mean(0) / dct['dd'][rc_, 0].mean()
    msk_ = np.isin(cf, codes); R_ = Rf[msk_]
    S_ = R_.T @ R_ / len(R_); S_ += lam * np.trace(S_) / 192 * np.eye(192)
    w_ = np.linalg.solve(S_, mu_); w_ /= w_ @ mu_
    return w_
def report(w_, drug, codes):
    out = []
    for (dr, fac), dct in EDR.items():
        gi_ = dct['gi']; rc_ = dct['reached'].astype(bool) & np.isin(cf[gi_], codes)
        r_ = dct['dX'][rc_] @ w_
        cfit = np.linalg.lstsq(np.c_[dct['dd'][rc_], np.ones(rc_.sum())], r_, rcond=None)[0]
        out.append(f"{dr[:3]}x{fac}: mean {r_.mean():.3f} dd-coef {np.round(cfit[:4],2)}")
    msk_ = np.isin(cf, codes)
    print(" | ".join(out), "| noise sd", np.round((Rf[msk_] @ w_).std(), 3))
for lam in [1e-3, 1e-2, 1e-1]:
    wI = mvdr_dec('insulin', [0, 1], lam); print("lam", lam); report(wI, 'insulin', [0, 1])
```