Run 10 shows a consistent causal insulin effect from the randomised within-tier rate. Each extra unit/h at hour t lowers glucose by about 1.9–2.0 mg/dL at t+1, 2.2–2.4 at t+2 and 1.8–1.9 at t+3. The numbers agree across tiers and between finger and lab readings. Glucose itself is very persistent (coefficient about 0.95–1.0).

So insulin truly lowers glucose, modestly per unit but persistently. Under ×2 this would accumulate over hours 13–24. These are total effects, though: they include policy feedback, where lower glucose means lower later tiers and less later insulin. The dose edit instead holds doses fixed, so I'll need to deconvolve that feedback later.

First I'll look at the norepinephrine policy to find its source of exogenous variation. I'll regress the rate on current MAP (arterial or cuff), the previous rate and the MAP change, then check whether the residual looks like iid noise.

```python
H['map_now'] = H.map_arterial.fillna(H.map_cuff)
H['map_src'] = np.where(H.map_arterial.notna(),'art',np.where(H.map_cuff.notna(),'cuff','none'))
gg = H.groupby('stay_id')
H['N_prev'] = gg.N.shift(1); H['map_prev'] = gg.map_now.shift(1)
H['map_last'] = gg.map_now.ffill()
ne = H[(H.N>0)].copy()
print('N>0 rows', len(ne), 'with map_now', ne.map_now.notna().mean(), ' N_prev>0 frac', (ne.N_prev>0).mean())
print(pd.crosstab(pd.cut(ne.map_now,[0,50,55,60,65,70,75,80,90,200]), pd.cut(ne.N_prev,[-0.01,0.0,0.2,0.4,0.6,0.8,0.99,1.0]), values=ne.N, aggfunc='mean').round(2).to_string())
# start of norepi: when N_prev==0
st = H[(H.N_prev==0)]
print(st.groupby(pd.cut(st.map_now,[0,50,55,60,65,70,75,80,90,200])).agg(p=('N',lambda v:(v>0).mean()), n=('N','size'), rate=('N',lambda v:v[v>0].mean())).to_string())
```