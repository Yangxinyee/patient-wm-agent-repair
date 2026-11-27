The thresholds are sharp on the recorded reading:
- At 179 the mean log-rate is 0.0 and at 181 it is 0.69.
- At exactly 180 it is 0.34, a 50/50 mix. That suggests the policy uses the unrounded measured value with a cut at 180.
- The same holds at 220 and 260.
- Below 150 there is never insulin.

The policy therefore acts on the measurement itself, not on hidden true glucose; otherwise measurement noise would blur the steps. Dosing then depends on the recorded history plus noise, and the noise is the candidate exogenous variation.

Next I'll test whether the noise is exogenous. With ε = log(rate) − log(base) at hours away from the thresholds, I'll check its correlation with:
- the previous hour's ε (persistence within the stay);
- the previous glucose change;
- glucose_lab minus finger at the same hour, as a proxy for hidden true glucose beyond the reading;
- MAP and norepinephrine;
- the stay's average ε (a stay-level component).

```python
s = P.copy()
s['base'] = base_ins(s.g_now.values) if True else None
s['eps'] = np.where(s.insulin.notna() & (s.base>0), np.log(s.insulin) - np.log(s.base.replace(0,np.nan)), np.nan)
near = s.g_now.isin([150,180,220,260,300])
s.loc[near, 'eps'] = np.nan
s['eps_prev'] = s.groupby('stay_id').eps.shift(1)
s['g_prev'] = s.groupby('stay_id').g_now.shift(1)
s['dg_past'] = s.g_now - s.g_prev
s['lab_minus_f'] = s.glucose_lab - s.glucose_finger
s['map_now'] = s.map_arterial.fillna(s.map_cuff)
cols = ['eps_prev','dg_past','lab_minus_f','map_now','norepinephrine','g_now','h']
for c in cols:
    m = s.eps.notna() & s[c].notna()
    print(c, m.sum(), round(np.corrcoef(s.eps[m], s[c][m])[0,1],4))
st = s.groupby('stay_id').eps.agg(['mean','count'])
st = st[st['count']>=10]
print("between-stay sd of mean eps:", st['mean'].std(), " expected if iid:", s.eps.std()/np.sqrt(st['count'].mean()))
```