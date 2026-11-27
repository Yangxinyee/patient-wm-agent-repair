The norepinephrine rule is a deterministic adjustment that depends on the MAP band and the previous rate:
- below 65 the rate goes up by roughly 0.03–0.08;
- at 65–75 it is held;
- above 75 it comes down by roughly 0.03;
- multiplicative noise (log sd ≈ 0.30) is applied, then the result is capped at 1.0.

At high previous rates the cap pulls the mean down, which explains the earlier oddities. The structure is "known function of observed history × random noise", which again gives a random innovation in each dose.

Next, the norepinephrine → MAP effect in stays with hourly arterial MAP. I'll regress arterial MAP at t+k on the rate at t, controlling flexibly for the previous rate and current MAP (band dummies plus linear terms). Because titration carries a dose innovation forward, I'll also estimate how much of the innovation persists in later rates. I'll do the same for insulin's possible effect on MAP, and norepinephrine's on glucose, so I know about cross-effects.

```python
P['m_now'] = P.map_arterial.fillna(P.map_cuff)
P['band'] = np.where(P.m_now<65,0,np.where(P.m_now<75,1,2))
g = P.groupby('stay_id')
for k in range(0,13):
    P[f'MA+{k}'] = g.map_arterial.shift(-k)
    P[f'NE+{k}'] = g.norepinephrine.shift(-k)
T = P[(P.ne_on==1)&(P.ne_prev_on==1)&P.m_now.notna()&(P.h<=23)].copy()
# flexible control: bins of prev rate x band
T['pbin'] = pd.cut(T.ne_prev, np.r_[0, np.quantile(T.ne_prev, np.linspace(0.05,0.95,19)), 1.01], duplicates='drop').cat.codes
T['cell'] = T.pbin*3 + T.band
cells = pd.get_dummies(T.cell, prefix='c', drop_first=False).astype(float)
res=[]
for k in range(1,13):
    mk = T[f'MA+{k}'].notna() & T.map_arterial.notna()
    sub = T[mk]; C = cells[mk].values
    X = np.column_stack([C, sub.map_arterial, sub.ne_prev, sub.norepinephrine])
    r,_ = ols(sub[f'MA+{k}'], X)
    mk2 = (T.h+k<=24)
    sub2 = T[mk2]; C2 = cells[mk2].values
    X2 = np.column_stack([C2, sub2.m_now, sub2.ne_prev, sub2.norepinephrine])
    r2,_ = ols(sub2[f'NE+{k}'].fillna(0), X2)
    res.append((k, mk.sum(), r.b.iloc[-1], r.se.iloc[-1], r2.b.iloc[-1], r2.se.iloc[-1]))
print(pd.DataFrame(res, columns=['k','n','MAP_eff','se','NE_persist','se2']).to_string())
```