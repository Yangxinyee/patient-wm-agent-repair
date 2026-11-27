The kernel still holds:
- `gg`, the training hourly grid with lagged readings, glucose band, expected insulin dose and its deviation;
- `gfm` and `mam`, the original model's teacher-forced predictions on 100 tuning stays, merged with doses.

Now I run the decisive check on the residuals (observed minus predicted mean):
- Estimate the policy's expected insulin rate as a function of the latest glucose, from the 30,000 training stays.
- Split each recorded rate into an expected part and a departure.
- Regress the residuals on both parts at lags 1–3.

Reading the result:
- If the model folded the missing effect into its glucose dynamics, the residuals will load on the departure (around −1.5 at lag 1) and not on the expected part.
- If they load equally on both, the model simply under-uses dose, and plain amplification would be correct.

```python
print([c_ for c_ in gg.columns])
import numpy as np, pandas as pd
G = gg[['stay_id','h','glu_now','ins']].copy(); G['ins']=G['ins'].fillna(0)
ok = G.glu_now.notna()
pf = G[ok].groupby(G[ok].glu_now.round()).ins.agg(['mean','size'])
pf['sm'] = pf['mean'].rolling(5, center=True, min_periods=1).mean()
pfun = lambda x: np.interp(x, pf.index.values, pf['sm'].values)
print(pf.loc[[140,149,150,160,179,180,200,219,220,249,250,275,299,300,350]].round(2).T)
# tuning grid with policy input glucose per hour
T2 = gt.copy().sort_values(['stay_id','h']); T2['ins']=T2['ins'].fillna(0)
T2['glu_now'] = T2['gf'].fillna(T2['gl'])
T2['p'] = np.where(T2.glu_now.notna(), pfun(T2.glu_now.fillna(0)), np.nan)
T2['e'] = T2['ins'] - T2['p']
for L_ in [1,2,3]:
    T2[f'e{L_}'] = T2.groupby('stay_id')['e'].shift(L_); T2[f'p{L_}'] = T2.groupby('stay_id')['p'].shift(L_); T2[f'd{L_}'] = T2.groupby('stay_id')['ins'].shift(L_)
q_ = gfm[['stay_id','h','observed','predicted_mean']].merge(T2[['stay_id','h']+[f'{a}{L_}' for a in 'epd' for L_ in [1,2,3]]], on=['stay_id','h'])
q_['r'] = q_.observed - q_.predicted_mean
qq = q_.dropna(subset=['e1','p1'])
import statsmodels.api as sm_ if False else None
from numpy.linalg import lstsq
for cols in [['e1','p1'], ['e1','e2','e3','p1','p2','p3'], ['d1','d2','d3']]:
    Z = qq[cols].fillna(0).values; Z = np.c_[np.ones(len(Z)), Z]
    for yv in ['r','predicted_mean','observed']:
        b_ = lstsq(Z, qq[yv].values, rcond=None)[0]; res_ = qq[yv].values - Z@b_
        se_ = np.sqrt(np.diag(np.linalg.inv(Z.T@Z))*res_.var())
        print(yv, cols, np.round(b_[1:],2), '+-', np.round(se_[1:],2))
print(len(qq), qq.r.std(), (qq.e1!=0).mean())
```