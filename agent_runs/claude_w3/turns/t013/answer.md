The measurement schedule is simple and known:
- After a policy glucose above 150, a finger reading is always taken the next hour.
- Otherwise a finger reading is taken with probability about 0.30, regardless of how long since the last one.
- A lab glucose is drawn in a random 20% of hours, independent of hour and glucose level.

So the lab readings give an outcome sampled at random, free of the treatment-dependent selection that affects finger readings once glucose falls below 150. Finger readings can also be used, with inverse-probability weights 1/0.3 when the last value was 150 or below.

Before using lab as the outcome, I need to know how lab and finger relate when taken in the same hour: the mean offset, the noise level, and whether the offset depends on the level. If the lab is the same underlying glucose with independent noise, effects on lab equal effects on finger. I will also look at the autocorrelation of lab readings over lags in untreated stays, to learn how persistent glucose deviations are, since persistence drives how one-hour effects accumulate.

```python
same = pn[(pn.glucose_finger.notna())&(pn.glucose_lab.notna())]
dd = same.glucose_lab - same.glucose_finger
print("n same-hour:", len(same), " lab-finger mean:", dd.mean().round(2), " sd:", dd.std().round(2))
lb = pd.cut(same.glucose_finger, [0,100,130,150,170,200,250,400])
print(same.assign(dd=dd).groupby(lb, observed=True).dd.agg(['size','mean','std']).round(2).to_string())
print("by lab level:", same.assign(dd=dd).groupby(pd.cut(same.glucose_lab,[0,100,130,150,170,200,250,400]), observed=True).dd.agg(['size','mean','std']).round(2).to_string())
# variance of finger noise: within-hour lab-finger var = var_f + var_l. Estimate var via lag-1 differences in untreated stays
never = st.index[st.any_ins==0]
nv = pn[pn.stay_id.isin(never)]
# lab autocorrelation at lags (lab-lab pairs) in never-insulin stays
labs = nv[nv.glucose_lab.notna()][['stay_id','h','glucose_lab']]
pairs = labs.merge(labs, on='stay_id', suffixes=('_a','_b'))
pairs = pairs[pairs.h_b>pairs.h_a]; pairs['lag']=pairs.h_b-pairs.h_a
res = pairs.groupby('lag').apply(lambda x: pd.Series({'n':len(x),'halfsqdiff':0.5*((x.glucose_lab_b-x.glucose_lab_a)**2).mean()}))
print(res.head(15).round(1).T.to_string())
fing = nv[nv.glucose_finger.notna()][['stay_id','h','glucose_finger']]
pf = fing.merge(fing, on='stay_id', suffixes=('_a','_b')); pf=pf[pf.h_b>pf.h_a]; pf['lag']=pf.h_b-pf.h_a
resf = pf.groupby('lag').apply(lambda x: pd.Series({'n':len(x),'halfsqdiff':0.5*((x.glucose_finger_b-x.glucose_finger_a)**2).mean()}))
print(resf.head(15).round(1).T.to_string())
print("var lab (never):", labs.glucose_lab.var().round(1), " var finger (never):", fing.glucose_finger.var().round(1))
```