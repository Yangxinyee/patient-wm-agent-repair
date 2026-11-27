fwd_3 holds the original model's factual pass on the 100 training stays: 5,684 readings with observed value, predicted mean and fine-bin NLL. I'll use it to see what the original model misses factually. For each glucose_finger and map_arterial reading I'll regress the residual (observed − predicted) on the drug rates at lags 1–3 and on their deviations from the policy. A clear negative insulin-deviation slope near −2, or a positive norepinephrine-innovation slope near +3.4, would show the effect is missing from the factual predictions. In that case adding it costs nothing and may even help.

```python
F3['res'] = F3['observed'] - F3['predicted_mean']
print(F3.groupby('event')['res'].agg(['count','mean','std']).round(2))
print(json.load(open('/work/fwd_3/summary.json')) if 'json' in globals() else open('/work/fwd_3/summary.json').read()[:800])
# per stay-hour drug rates from recs (infusions start at hour+5min; 'hour' = integer start hour)
dr = recs[recs.event.isin(['insulin','norepinephrine'])].copy(); dr['h'] = np.floor(dr.hour).astype(int)
RI = dr[dr.event=='insulin'].groupby(['stay_id','h']).rate.sum(); RN = dr[dr.event=='norepinephrine'].groupby(['stay_id','h']).rate.sum()
def lagrate(Ser, s, h): return float(Ser.get((s, h), 0.0))
for ev, Ser in [('glucose_finger', RI), ('map_arterial', RN)]:
    T_ = F3[F3.event==ev].copy(); hh = np.floor(T_.hour).astype(int).values; ss = T_.stay_id.values
    for L in (1,2,3): T_[f'r{L}'] = [lagrate(Ser, s, h-L) for s, h in zip(ss, hh)]
    import statsmodels.api as sm if False else None
    X_ = np.column_stack([np.ones(len(T_)), T_[['r1','r2','r3']].values]); b_, *_ = np.linalg.lstsq(X_, T_.res.values, rcond=None)
    print(ev, 'n', len(T_), 'res ~ r1,r2,r3 coefs', b_[1:].round(3))
```