Lab and finger glucose measure the same underlying glucose. Their same-hour difference averages 0.04 with sd 9. The noise grows with the level, roughly in proportion, and the finger is the noisier of the two.

In stays that never got insulin, the half squared difference between readings L hours apart keeps rising by about 8 (mg/dL)² per hour for at least 15 hours. Glucose deviations are therefore nearly permanent within a stay's time span: it behaves close to a random walk rather than snapping back to a set point. If insulin's effect enters glucose the same way, a one-hour decrease would persist, and doubling insulin for 12 hours could add up to a large shift. If instead insulin acts transiently, the shift would stay small.

This needs a direct estimate of the impulse response with later doses held fixed. I will run a joint IV regression for each lag L = 1..12, with origins at hours where the policy glucose exceeds 150:
- outcome: the lab glucose at t+L (randomly sampled, so no selection);
- endogenous regressors: the doses d_t … d_{t+L−1};
- instruments: the jitter residuals z_s = d_s − E[d | policy glucose].

The coefficient on d_t at each L is the lag-L effect of a unit dose. A persistent, roughly constant negative value of about −2 would confirm the integrator picture. Coefficients decaying to zero would refute it. A coefficient growing beyond −2 would mean delayed action.

```python
# arrays per stay, hours 0..24
H = 25
pn0 = pn[(pn.h>=0)&(pn.h<=24)].sort_values(['stay_id','h'])
sids = pn0.stay_id.unique(); S = len(sids)
def arr(col): return pn0[col].values.reshape(S, H)
GF = arr('glucose_finger'); GL = arr('glucose_lab'); INS = arr('insulin')
fresh_f = ~np.isnan(GF); fresh_l = ~np.isnan(GL)
POL = np.where(fresh_f, GF, np.where(fresh_l, GL, np.nan))
# m(pol) = E[d | pol] for fresh readings
pv = POL[~np.isnan(POL)]; dv = INS[~np.isnan(POL)]
mtab = pd.Series(dv).groupby(np.round(pv)).mean()
mtab_full = mtab.reindex(np.arange(0, 600)).fillna(0)
# smooth for sparse high values: use band-based mean for pol>300
M = np.where(np.isnan(POL), 0.0, mtab_full.values[np.clip(np.nan_to_num(np.round(POL)).astype(int),0,599)])
Z = np.where(np.isnan(POL), 0.0, INS - M)
print("check: mean z by fresh pol>150:", Z[POL>150].mean().round(4), " sd z:", Z[POL>150].std().round(3))
np.savez('/work/arrays.npz', GF=GF, GL=GL, INS=INS, POL=POL, Z=Z, sids=sids)

def iv_lag(L, Y=GL, ctrl_bin=2):
    rows = []
    for t in range(0, H-L):
        ok = (POL[:,t]>150) & ~np.isnan(Y[:,t+L])
        if ok.sum()==0: continue
        rows.append(np.column_stack([np.full(ok.sum(), t), POL[ok,t], Y[ok,t+L], INS[ok, t:t+L], Z[ok, t:t+L]]))
    A = np.vstack(rows)
    tcol, pol, y = A[:,0], A[:,1], A[:,2]
    Dm = A[:,3:3+L]; Zm = A[:,3+L:3+2*L]
    # FE for (t, pol bin): demean
    key = pd.Series(list(zip(tcol.astype(int), (pol//ctrl_bin).astype(int))))
    df = pd.DataFrame(np.column_stack([y, Dm, Zm])); grp = key.factorize()[0]
    dm = df - df.groupby(grp).transform('mean')
    yv = dm.iloc[:,0].values; Dv = dm.iloc[:,1:1+L].values; Zv = dm.iloc[:,1+L:].values
    # 2SLS
    PZ = Zv @ np.linalg.lstsq(Zv, Dv, rcond=None)[0]
    b = np.linalg.lstsq(PZ, yv, rcond=None)[0]
    res = yv - Dv@b
    V = np.linalg.inv(PZ.T@PZ) * res.var()
    return b, np.sqrt(np.diag(V)), len(yv)
for L in [1,2,3,4,6,8,10,12]:
    b, se, n = iv_lag(L)
    print(f"L={L:2d} n={n:6d} coef on d_t (lag L): {b[0]:7.3f} ({se[0]:.3f});  coefs by lag (L..1): ", np.round(b,2))
```