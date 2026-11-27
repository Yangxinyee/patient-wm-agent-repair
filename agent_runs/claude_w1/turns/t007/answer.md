Insulin runs as one unbroken block in only about half of treated stays, so it is often paused and restarted. Norepinephrine is unbroken in 88% of treated stays. Most treated stays start at hour 0.

Next I'll characterise the insulin policy: P(on) and the rate, as functions of the latest glucose reading and the previous hour's rate. I'll also check whether a fresh reading at hour h matters. That shows how much dose variation is noise conditional on observables.

```python
P = Ptr.reset_index().sort_values(['stay_id','h'])
g = P.groupby('stay_id')
for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    P['last_'+c] = g[c].ffill()
P['ins_prev'] = g.insulin.shift(1)
P['ne_prev'] = g.norepinephrine.shift(1)
P['ins_on'] = P.insulin.notna().astype(int)
P['ne_on'] = P.norepinephrine.notna().astype(int)
P['ins_prev_on'] = P.ins_prev.notna().astype(int)
P['ne_prev_on'] = P.ne_prev.notna().astype(int)
# insulin-ever stays
insever = P.groupby('stay_id').ins_on.transform('max')==1
Q = P[insever & (P.h>=0)]
Q['gbin'] = pd.cut(Q.last_glucose_finger, [0,100,120,140,160,180,200,220,250,300,1000])
print(Q.groupby(['ins_prev_on','gbin']).ins_on.agg(['mean','size']).to_string())
print(Q[Q.ins_on==1].groupby(['ins_prev_on','gbin']).insulin.agg(['mean','std','size']).to_string())
```