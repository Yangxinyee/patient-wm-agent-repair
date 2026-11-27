Alignment check: `Dm` holds insulin doses and `Rm` norepinephrine. Column h holds the dose that is lag 1 for a reading at hour h, i.e. the dose given at hour h−1. `keep` lines up with `rs_all`/`rt_all` exactly.

Next I regress the replica's factual residuals on lagged rule innovations:
- glucose_finger residuals on insulin innovations `NUm`;
- MAP_arterial residuals on norepinephrine innovations `NUNm`;
- 8 lags, standard errors clustered by stay.

These coefficients are the c_k, the direct dose effect the model is missing given its history. For comparison I'll also show the coefficients on raw dose lags. If the model under-uses doses, the innovation coefficients will be clearly non-zero (negative for insulin, positive for norepinephrine). The raw-dose coefficients will be distorted by the model's compensation through the reading.

```python
def lagmat(M, ix, hr, K):
    out = np.zeros((len(ix), K))
    for k in range(1, K+1):
        col = (hr - k + 1).astype(int); ok = (col >= 0) & (col < M.shape[1])
        out[ok, k-1] = M[ix[ok], col[ok]]
    return out
def ols_cl(y, X, cl):
    X = np.c_[np.ones(len(y)), X]; XtX = np.linalg.inv(X.T @ X); bcoef = XtX @ X.T @ y; r = y - X @ bcoef
    dfc = pd.DataFrame(X * r[:, None]); dfc['cl'] = cl; Sg_ = dfc.groupby('cl').sum().values
    V_ = XtX @ (Sg_.T @ Sg_) @ XtX; return bcoef, np.sqrt(np.diag(V_))
K_ = 8
for cd_, NU_, D_ in [('glucose_finger', NUm, Dm), ('map_arterial', NUNm, Rm)]:
    sel = (codesK == cd_) & (hrK >= 1)
    Ln_ = lagmat(NU_, ixK[sel], hrK[sel], K_); Ld_ = lagmat(D_, ixK[sel], hrK[sel], K_)
    print(cd_, 'n', sel.sum(), 'corr(nu1,d1)', np.corrcoef(Ln_[:,0], Ld_[:,0])[0,1].round(3), 'sd nu', Ln_.std(0)[:3].round(3))
    b_, se_ = ols_cl(res_all[sel], Ln_, ixK[sel])
    print('  res~nu  ', b_[1:].round(3).tolist(), ' se', se_[1:].round(3).tolist())
    b_, se_ = ols_cl(res_all[sel], Ld_, ixK[sel])
    print('  res~dose', b_[1:].round(3).tolist(), ' se', se_[1:].round(3).tolist())
```