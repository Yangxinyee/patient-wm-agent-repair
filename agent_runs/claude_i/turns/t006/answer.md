The examples show different patterns for the two drugs:
- **Insulin** is intermittent and low-rate, roughly 0.5–2 u/h. It switches on and off and only loosely tracks the latest glucose.
- **Norepinephrine** usually runs continuously once started and is titrated.

Some stays have only arterial MAP and others only cuff MAP. To find what drives dosing, I'll tabulate how the insulin decision depends on the readings, both the probability of giving it and the rate chosen. I'll use the glucose at hour h, the latest prior glucose, and the previous rate. I'll do the same for norepinephrine against MAP. I'll also check the stay_id structure and hospital-admission times, since those could encode a confounder or a usable source of variation.

```python
def last_obs(P, col):
    # last observed value at or before hour h within stay, and its age
    v = P[col].groupby(level='stay_id').ffill()
    t = pd.Series(np.where(P[col].notna(), P.index.get_level_values('h'), np.nan), index=P.index).groupby(level='stay_id').ffill()
    return v, P.index.get_level_values('h') - t
P['g_any'] = P.gf.combine_first(P.gl)
P['g_last'], P['g_age'] = last_obs(P,'g_any')
P['m_any'] = P.ma.combine_first(P.mc)
P['m_last'], P['m_age'] = last_obs(P,'m_any')
P['ins0'] = P.ins.fillna(0); P['ne0']=P.ne.fillna(0)
P['ins_prev'] = P.groupby(level='stay_id').ins0.shift(1); P['ne_prev']=P.groupby(level='stay_id').ne0.shift(1)
Q = P[P.index.get_level_values('h')>=0]
gb = pd.cut(Q.g_last, [0,80,100,110,120,130,140,150,160,170,180,200,220,250,300,500])
print(Q.groupby(gb, observed=True).agg(n=('ins0','size'), p_ins=('ins', lambda x: x.notna().mean()), mean_rate=('ins','mean')).round(3).to_string())
mb = pd.cut(Q.m_last, [0,40,50,55,60,65,70,75,80,85,90,100,120,200])
print(Q.groupby(mb, observed=True).agg(n=('ne0','size'), p_ne=('ne', lambda x: x.notna().mean()), mean_rate=('ne','mean')).round(3).to_string())
hosp = train[train.event=='hospital_admission'][['stay_id','hour']].set_index('stay_id').hour
print("hosp admission hour dist:", hosp.describe(), hosp.round(1).value_counts().head(10).to_dict())
print("stay ids sample:", sorted(train.stay_id.unique())[:3], sorted(train.stay_id.unique())[-3:], sorted(tune.stay_id.unique())[:3], sorted(tune.stay_id.unique())[-3:])
```