The norepinephrine residual passes the same exogeneity checks as insulin. Its correlation with each of these is at most 0.007 in absolute value:
- the previous residual;
- the past MAP change;
- cuff − arterial;
- glucose, the hour, and current MAP.

The rate's deviation from the policy's expected value, given the previous rate and the MAP category, is therefore randomised.

Next I'll g-estimate the per-dose effect of norepinephrine on arterial MAP at lags 1–12, with later doses held fixed, using the same linear structural nested mean model. I'll regression-adjust the outcome on current and previous MAP and the previous rate.

Units: γ_j is the MAP change per 1 mcg/kg/min for one hour, and rates are about 0.5. A clearly positive γ would mean norepinephrine raises MAP. γ near zero would mean the model's dose response is purely confounded and the true effect is absent.

```python
def gest2(Y, R, Rres, use, covs, maxlag=12):
    nS, nH = R.shape
    gam = np.zeros(maxlag+1); A = np.zeros((maxlag+1, maxlag+1)); b = np.zeros(maxlag+1)
    for l in range(1, maxlag+1):
        for s_ in range(1, nH-l):
            y = Y[:, s_+l]; m = ~np.isnan(y) & use[:, s_]
            if m.sum() < 50: continue
            r = Rres[m, s_]; yy = y[m]
            X = np.column_stack([np.ones(m.sum())] + [np.nan_to_num(c[m, s_], nan=np.nanmean(c[:, s_])) for c in covs])
            cb, *_ = np.linalg.lstsq(X, yy, rcond=None); yy = yy - X@cb
            b[l] += (r*yy).sum()
            for j in range(1, l+1):
                A[l, j] += (r*R[m, s_+l-j]).sum()
        gam[l] = (b[l] - (A[l,1:l]*gam[1:l]).sum())/A[l,l]
    return gam, A, b
Rn = np.nan_to_num(N_)
Mprev = np.full_like(Mn, np.nan); Mprev[:,1:] = Mn[:,:-1]
cm_grid = np.zeros_like(N_); cm_grid[given] = cellmean
gam_ne, A_ne, b_ne = gest2(A_, Rn, Nres, Nuse, [Mn, Mprev, np.nan_to_num(Np), cm_grid])
print("NE->MAP art per 1 mcg/kg/min:", np.round(gam_ne[1:],2), "cum", round(gam_ne[1:].sum(),2))
gam_ne_c, *_ = gest2(C_, Rn, Nres, Nuse, [Mn, Mprev, np.nan_to_num(Np), cm_grid])
print("NE->MAP cuff:", np.round(gam_ne_c[1:],2), "cum", round(gam_ne_c[1:].sum(),2))
```