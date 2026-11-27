That run failed on a stray import line I left in by mistake. This is the same analysis without it:
- fit the policy-expected insulin rate from training records;
- split recorded doses into an expected part and a departure;
- regress the original model's residuals, its predicted means and the observed readings on both parts.

The same reading applies. Residuals loading only on the departure would mean the model folded the missing effect into its glucose dynamics.

```python
print([c_ for c_ in gg.columns])
G = gg[['stay_id','h','glu_now','ins']].copy(); G['ins']=G['ins'].fillna(0)
ok = G.glu_now.notna()
pf = G[ok].groupby(G[ok].glu_now.round()).ins.agg(['mean','size'])
pf['sm'] = pf['mean'].rolling(5, center=True, min_periods=1).mean()
pfun = lambda x: np.interp(x, pf.index.values, pf['sm'].values)
print(pf.reindex([140,149,150,160,179,180,200,219,220,249,250,275,299,300,350]).round(2).T)
T2 = gt.copy().sort_values(['stay_id','h']); T2['ins']=T2['ins'].fillna(0)
T2['glu_now'] = T2['gf'].fillna(T2['gl'])
T2['p'] = np.where(T2.glu_now.notna(), pfun(T2.glu_now.fillna(0)), np.nan)
T2['e'] = T2['ins'] - T2['p']
for L_ in [1,2,3]:
    T2[f'e{L_}'] = T2.groupby('stay_id')['e'].shift(L_); T2[f'p{L_}'] = T2.groupby('stay_id')['p'].shift(L_); T2[f'd{L_}'] = T2.groupby('stay_id')['ins'].shift(L_)
q_ = gfm[['stay_id','h','observed','predicted_mean']].merge(T2[['stay_id','h']+[f'{a}{L_}' for a in 'epd' for L_ in [1,2,3]]], on=['stay_id','h'])
q_['r'] = q_.observed - q_.predicted_mean
qq = q_.dropna(subset=['e1','p1'])
from numpy.linalg import lstsq
for cols in [['e1','p1'], ['e1','e2','e3','p1','p2','p3'], ['d1','d2','d3']]:
    Z = qq[cols].fillna(0).values; Z = np.c_[np.ones(len(Z)), Z]
    for yv in ['r','predicted_mean','observed']:
        b_ = lstsq(Z, qq[yv].values, rcond=None)[0]; res_ = qq[yv].values - Z@b_
        se_ = np.sqrt(np.diag(np.linalg.inv(Z.T@Z))*res_.var())
        print(yv, cols, np.round(b_[1:],2), '+-', np.round(se_[1:],2))
print(len(qq), qq.r.std(), (qq.e1!=0).mean())
```