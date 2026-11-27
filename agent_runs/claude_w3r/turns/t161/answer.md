Factual costs (Gaussian approximation, nats per reading over all readings; negative means better):
- Insulin in deviation form with band-mean compensation, b = (−2.0, −0.95, −0.26): −0.0008, i.e. slightly better. Uncompensated levels: +0.026, too costly.
- Norepinephrine levels without an intercept: +0.02 to +0.08, too costly. The terms add a mean shift, and the original model has absorbed part of the drug's association through its MAP dynamics.

So insulin is settled in deviation form. Norepinephrine needs compensation from information the head has: the last MAP value (register), plus an intercept, and r2 from the END atom. Next I'll fit OLS of the MAP residuals on [1, r1, r2, r3, last MAP] and price fixed-β variants with the other coefficients refit. Cheap fits (≤ 0) with β1 near 3.4 would confirm the plan is viable.

```python
M_ = recs[recs.event=='map_arterial'].copy(); M_['h']=np.floor(M_.hour).astype(int); mapv = M_.groupby(['stay_id','h']).value.last()
def lastval(s, h, ser):
    for L in range(1, 6):
        v = ser.get((s, h-L), np.nan)
        if v == v: return v, L
    return np.nan, 9
lv_ = [lastval(s, int(np.floor(h)), mapv) for s, h in zip(TM.stay_id, TM.hour)]
TM['mlast'] = [a for a, _ in lv_]; TM = TM[TM.mlast.notna()].copy(); TM['mc'] = TM.mlast - 75
def fitd(T_, cols, fixed=None, s2=None):
    y_ = T_.res.values.copy(); fixed = fixed or {}
    for c, b in fixed.items(): y_ = y_ - b*T_[c].values
    X_ = np.column_stack([np.ones(len(T_))] + [T_[c].values for c in cols]); b_, *_ = np.linalg.lstsq(X_, y_, rcond=None)
    d_ = X_ @ b_ + sum(b*T_[c].values for c, b in fixed.items())
    return b_.round(3), round(dnll(T_, d_, s2),5)
print('free r1,r2,r3,mc', fitd(TM, ['r1','r2','r3','mc'], s2=s2m))
print('free r1,r2,mc', fitd(TM, ['r1','r2','mc'], s2=s2m))
for b1_, b2_ in [(3.45,0.72),(3.45,0),(5,1),(6,1.5)]:
    print('fix', b1_, b2_, fitd(TM, ['mc','r3'], fixed={'r1':b1_,'r2':b2_}, s2=s2m))
```