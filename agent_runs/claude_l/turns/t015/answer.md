The dosing noise is clean, and insulin has a modest but real effect.
- **Noise is iid:** the policy residual has essentially no autocorrelation within a stay (0.004), and stay-mean residuals are no more spread than iid noise predicts (0.142 vs 0.160). So it is independent hour-to-hour variation, not a clinician or patient style. That makes it a credible instrument.
- **One-hour effect:** a 1 U/h higher insulin rate lowers the next hour's finger glucose by 2.06 mg/dL (SE 0.045). The estimate is stable across subsets: all rows, on insulin, glucose > 160.
- **Persistence:** the lag coefficients (−0.39 on g, +0.40 on g_prev) suggest glucose is very persistent, with noise on top. If so, the evaluation's open-loop effect over hours 13–24 could build up well beyond 2 mg/dL.

Before working out the dynamics I'll run the same analysis for norepinephrine on MAP:
1. Fit the NE policy and check whether its residual is clean. The negative control is same-hour cuff MAP given arterial MAP.
2. Estimate the one-hour effect of the NE residual on the next arterial MAP.

A coefficient near zero would mean NE does nothing in this ICU. A clear positive one would mean it raises MAP.

```python
d['ma_prev'] = d.groupby('stay_id').map_arterial.shift(1)
d['ma_prev2'] = d.groupby('stay_id').map_arterial.shift(2)
d['mc_prev_last'] = d.groupby('stay_id').mc_last.shift(1)
d['g_last_prev'] = d.groupby('stay_id').g_last.shift(1)
Xn = ['map_arterial','ma_prev','ma_prev2','ne_prev','mc_last','mc_prev_last','ih','ins_prev','g_last']
n_ = d[d.map_arterial.notna()].copy()
itr = (n_.stay_id % 5 != 0).values
gbn = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1).fit(n_.loc[itr, Xn], n_.loc[itr,'norepinephrine'])
n_['ne_hat'] = gbn.predict(n_[Xn]); n_['nres'] = n_.norepinephrine - n_.ne_hat
print("NE R2 holdout:", 1-n_.nres[~itr].var()/n_.norepinephrine[~itr].var(), "resid sd", n_.nres[~itr].std(), "sd among NE>0:", n_.nres[(~itr)&(n_.norepinephrine>0)].std())
# NE policy shape
n_['mbin'] = pd.cut(n_.map_arterial,[0,50,55,60,65,70,75,80,90,200])
print(n_.groupby('mbin').norepinephrine.agg(['mean','std',lambda x:(x>0).mean(),'count']).round(3))
# negative control: same-hour cuff
nc = n_[n_.map_cuff.notna() & (~itr)]
print("cuff_t ~ art_t, ne_hat, nres:", *[x.round(3) for x in ols(nc.map_cuff.values, [nc.map_arterial, nc.ne_hat, nc.nres])])
# effect on next arterial MAP
n_ = n_.sort_values(['stay_id','ih'])
n_['m_next'] = n_.groupby('stay_id').map_arterial.shift(-1); n_['ih_next'] = n_.groupby('stay_id').ih.shift(-1)
qn = n_[(n_.ih_next==n_.ih+1) & n_.ma_prev.notna()].copy(); qn['dm'] = qn.m_next - qn.map_arterial
for lab, sub in [('all', qn), ('ne>0', qn[qn.norepinephrine>0])]:
    print(lab, len(sub), *[x.round(3) for x in ols(sub.dm.values, [sub.map_arterial, sub.ma_prev, sub.ne_hat, sub.nres])])
```