The g-estimation gives per-dose effects with later doses held fixed. One extra unit-hour of insulin at hour s changes glucose at s + j by:

| lag j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11+ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| γ_j (mg/dL) | −2.0 | −2.4 | −2.2 | −1.7 | −1.4 | −1.1 | −0.5 | −0.4 | −0.4 | −0.2 | ≈0 |

The effect builds over 2–3 hours and fades over about 8–10, a delayed pharmacodynamic response. The cumulative effect of a sustained extra 1 u/h is about −12 mg/dL. For a stay on about 2 u/h, doubling from hour 12 would lower glucose increasingly, toward about −24 by the end. The 13–24 average might be around −10 to −15 depending on how doses are distributed.

These estimates were not variance-reduced. Next I'll re-run the g-estimation with regression adjustment of the outcome on history, which reduces variance without adding bias. I'll also check linearity and effect modification by estimating separately by glucose band (base 1, 2, 3+). Similar per-unit effects across bands would support a linear model, so that x2 and x1/2 give symmetric shifts (+1× and −0.5× the dose). Per-unit effects that fall with band would point to saturation or effects proportional to glucose.

```python
def gest(F, R, Rres, use, maxlag=12, adjust=True, Gn=None):
    nS, nH = R.shape
    gam = np.zeros(maxlag+1); A = np.zeros((maxlag+1, maxlag+1)); b = np.zeros(maxlag+1)
    for l in range(1, maxlag+1):
        for s_ in range(1, nH-l):
            y = F[:, s_+l]; m = ~np.isnan(y) & use[:, s_]
            if m.sum() < 50: continue
            r = Rres[m, s_]; yy = y[m]
            if adjust:
                g0 = Gn[m, s_]; g1 = np.nan_to_num(Gn[m, s_-1], nan=np.nanmean(Gn[:, s_-1])); r1 = R[m, s_-1]
                X = np.column_stack([np.ones(m.sum()), g0, g0**2/100, g1, r1, R[m, s_]-Rres[m, s_]])
                cb, *_ = np.linalg.lstsq(X, yy, rcond=None); yy = yy - X@cb
            b[l] += (r*yy).sum()
            for j in range(1, l+1):
                A[l, j] += (r*R[m, s_+l-j]).sum()
        gam[l] = (b[l] - (A[l,1:l]*gam[1:l]).sum())/A[l,l]
    return gam
use_all = ins_given & ~np.isnan(Gn)
gam_all = gest(F, R, Rres, use_all, Gn=Gn)
print("all:", np.round(gam_all[1:],3), "cum", np.round(gam_all[1:].sum(),2))
bands = {'b1(150-180)': (Gn>=150)&(Gn<180), 'b2(180-220)': (Gn>180)&(Gn<220), 'b3+(>220)': (Gn>220)}
for k_, msk in bands.items():
    g_ = gest(F, R, Rres, use_all & msk, Gn=Gn)
    print(k_, np.round(g_[1:],2), "cum", np.round(g_[1:].sum(),2))
```