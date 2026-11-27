Run 25 shows the bundle layout. Readings come in a bundle at h:00 and drugs in a bundle at h:05. Drug bundles hold up to 4 atoms (insulin and norepinephrine starts plus the END atoms of the previous hour's records). So the next glucose prediction at t+1 is made from the state after the h:05 drug bundle.

Next I'll test whether that state carries the dose, using cross-validated ridge regressions on the 992-dim state:
- the insulin rate I(t) at insulin bundles, and its within-tier deviation from the tier mean, which is the part the model ignores for glucose;
- the norepinephrine rate, and its deviation from the previous rate.

A high R² for the deviation means a linear readout of the state can carry the missing causal signal to the heads. A low R² means the dose information is lost early, so the edit must act at the input or core level.

```python
X = S['x'].astype(np.float64)
st['h'] = np.floor(st.hour+1e-9).astype(int)
st['minute'] = np.round((st.hour-np.floor(st.hour))*60).astype(int)
recH = H.set_index(['stay_id','h'])
dr = st[(st.minute==5)].copy()
dr = dr.join(recH[['I','N','tier','glucose_finger','N_prev','map_now']], on=['stay_id','h'])
dr['idx'] = dr.index
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, GroupKFold
def cvr2(Xs, y, groups):
    m = RidgeCV(alphas=np.logspace(-2,4,13))
    p = cross_val_predict(m, Xs, y, cv=GroupKFold(5), groups=groups)
    return 1-((y-p)**2).sum()/((y-y.mean())**2).sum(), p
di = dr[(dr.ins)&(dr.tier>=1)].copy()
tier_mean = H[(H.I>0)&(H.tier>=1)].groupby('tier').I.mean()
di['I_dev'] = di.I - di.tier.map(tier_mean)
for nm, y in [('I', di.I.values), ('I_dev', di.I_dev.values)]:
    r2, p = cvr2(X[di.idx.values], y, di.stay_id.values)
    print('insulin', nm, 'n', len(y), 'cv R2 from state', round(r2,3), 'slope', round(np.polyfit(p, y, 1)[0],3))
dn = dr[(dr.ne)&(dr.N>0)&(dr.N_prev>0)].copy()
dn['N_dev'] = np.log(dn.N/dn.N_prev)
for nm, y in [('N', dn.N.values), ('logN_ratio', dn.N_dev.values)]:
    r2, p = cvr2(X[dn.idx.values], y, dn.stay_id.values)
    print('norepi', nm, 'n', len(y), 'cv R2 from state', round(r2,3))
```