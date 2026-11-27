Hours −1 to 24 carry readings and hours 0 to 24 carry drug records. Each hour has a glucose finger-stick about 52–59% of the time and a lab glucose about 20% of the time. Insulin runs in 31–42% of stay-hours, rising over the stay; norepinephrine runs in about 32%. Norepinephrine is hard-capped at 1.0 mcg/kg/min (13.6% of records), so "norepinephrine ×2" pushes many doses well beyond anything in training. The key question is the dosing policy. If doses depended only on recorded readings, a well-fit model would already answer correctly. The task implies it does not, so I expect hidden confounding or a policy with exploitable structure: thresholds, noise, or stale information.

Next I will look at:
- per-stay drug patterns (who gets insulin, in how many hours);
- the joint distribution of the insulin rate at hour h with the glucose at hour h, and of the norepinephrine rate with MAP;
- how deterministic the policy is.

```python
P = Ptr.copy()
st_ins = P.insulin.notna().groupby(level=0).mean()
st_ne = P.norepinephrine.notna().groupby(level=0).mean()
print("frac hours w/ insulin per stay:\n", pd.cut(st_ins,[-0.01,0,0.1,0.3,0.5,0.7,0.9,1.0]).value_counts().sort_index())
print("frac hours w/ NE per stay:\n", pd.cut(st_ne,[-0.01,0,0.1,0.3,0.5,0.7,0.9,1.0]).value_counts().sort_index())
print(pd.crosstab(st_ins>0, st_ne>0))
# insulin rate vs glucose_finger same hour
d = P[P.glucose_finger.notna()].copy()
d['gbin'] = pd.cut(d.glucose_finger, [0,80,100,120,140,160,180,200,220,250,300,400,1000])
print(d.groupby('gbin').agg(n=('glucose_finger','size'), p_ins=('insulin', lambda x: x.notna().mean()), mean_rate=('insulin','mean'), sd_rate=('insulin','std')))
d2 = P[P.map_arterial.notna()].copy()
d2['mbin'] = pd.cut(d2.map_arterial, [0,40,50,55,60,65,70,75,80,90,100,200])
print(d2.groupby('mbin').agg(n=('map_arterial','size'), p_ne=('norepinephrine', lambda x: x.notna().mean()), mean_rate=('norepinephrine','mean'), sd_rate=('norepinephrine','std')))
```