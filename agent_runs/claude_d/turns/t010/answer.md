The insulin trigger is sharp. Insulin is started that hour once glucose_finger is about 150 or higher (given 96–97% of the time, about 3% omitted). The rate steps up at 180, and below 150 nothing is given. When there is no finger reading that hour, insulin is rare (2.4%), and those cases seem to coincide with hours that have only a lab reading (11% given). So the policy probably falls back on glucose_lab.

Next I will:
- write the insulin rule as step(G), using finger or else lab;
- compute the multiplier = rate / step;
- test whether the multiplier persists within a stay (a stay-level aggressiveness) or is fresh each hour;
- test whether it relates to the patient's own glucose dynamics, which would mean confounding.

Hour-level independent noise would give a clean natural experiment. A stay-level multiplier would need a check against patient traits.

```python
def step_of(G):
    G = np.asarray(G, float)
    return np.select([G<150, G<180, G<220, G<260, G<300], [0,1,2,3,4], default=6).astype(float)
cur = P[P.gf.notna()].copy()
cur['base'] = step_of(cur.gf)
print(cur.groupby(pd.cut(cur.gf,[150,180,220,260,300,320,340,360,400]),observed=True).ins.agg(['mean','std','size']))
c1 = cur[(cur.base>0)&cur.ins.notna()].copy(); c1['m']=c1.ins/c1.base
print(c1.groupby('base').m.describe())
print(c1[c1.base==2].ins.value_counts().sort_index())
print(c1[c1.base==1].ins.value_counts().sort_index())
# lab-only hours
lo = P[P.gf.isna() & P.gl.notna()].copy(); lo['base']=step_of(lo.gl)
print(lo.groupby('base').agg(n=('ins0','size'), p=('ins',lambda x: x.notna().mean()), r=('ins','mean')))
# within-stay persistence of multiplier
c1['res'] = np.log(c1.ins) - np.log(c1.base)
sm = c1.groupby(level='stay_id').res.agg(['mean','size','var'])
sm = sm[sm['size']>=4]
print('between-stay var of mean res', sm['mean'].var(), 'mean within var', sm['var'].mean(), 'mean n', sm['size'].mean())
```