Among running, uncapped infusions the rate change depends mainly on the previous rates. The fit is r_t ≈ 0.27 + 0.74·r_{t−1} + 0.036·r_{t−2} − 0.003·MAP_t, i.e. mean reversion toward a MAP-dependent target. Once previous rates are included, MAP_{t−1} adds nothing (R² of Δr is about 0.19, the rest is innovation). The placebo failure may therefore come from my too-coarse policy model (missing r_{t−2} and the AR structure), not from hidden confounding.

Next I will fit a flexible, cross-fitted policy model for the norepinephrine rate, with gradient boosting on MAP lags 0–2, rate lags 1–3, cuff MAP and hour, to get the innovation x_t. I will then run a placebo on variables the policy model does not use (MAP at t−3 and t−4). If the placebos pass, the innovation is as-if random.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
def lagcol(Ar, s_idx, t_idx, L):
    tt = t_idx-L; out = np.full(len(s_idx), np.nan); ok = tt>=0; out[ok]=Ar[s_idx[ok], tt[ok]]; return out
rows=[]
for t in range(1,T):
    m_ok = ~np.isnan(ma_full[:,t])
    s_idx=np.where(m_ok)[0]; t_idx=np.full(len(s_idx),t)
    d=pd.DataFrame({'stay':s_idx,'t':t,'r':ne_r[s_idx,t]})
    for L in range(0,5): d['m%d'%L]=lagcol(ma_full,s_idx,t_idx,L)
    for L in range(1,4): d['r%d'%L]=lagcol(ne_r,s_idx,t_idx,L)
    d['mc0']=lagcol(mc,s_idx,t_idx,0)
    rows.append(d)
NP = pd.concat(rows, ignore_index=True)
feats=['m0','m1','m2','r1','r2','r3','mc0','t']
NP['rhat']=np.nan; fold = NP.stay%2
for k in [0,1]:
    mdl=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63)
    mdl.fit(NP.loc[fold!=k,feats], NP.loc[fold!=k,'r']); NP.loc[fold==k,'rhat']=mdl.predict(NP.loc[fold==k,feats])
NP['x']=NP.r-NP.rhat
print('R2 policy', 1-NP.x.var()/NP.r.var(), 'innovation sd', NP.x.std())
NP['m02']=(NP.m0-70)**2/100
for pl in ['m3','m4']:
    b,s_,nn=ols(NP,pl,['x','m0','m02','m1','m2','r1','r2']); print('placebo',pl,'coef %.3f se %.3f n %d'%(b.x,s_.x,nn))
```