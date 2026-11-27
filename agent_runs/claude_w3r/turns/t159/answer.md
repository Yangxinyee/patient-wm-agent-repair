A syntax slip: a stray line. I'm removing it and rerunning the same residual analysis. The hypothesis is unchanged: if the factual residuals have a negative slope on recent insulin and a positive slope on recent norepinephrine, those drug effects are missing from the original predictions.

```python
F3['res'] = F3['observed'] - F3['predicted_mean']
print(F3.groupby('event')['res'].agg(['count','mean','std']).round(2))
print(open('/work/fwd_3/summary.json').read()[:600])
dr = recs[recs.event.isin(['insulin','norepinephrine'])].copy(); dr['h'] = np.floor(dr.hour).astype(int)
RI = dr[dr.event=='insulin'].groupby(['stay_id','h']).rate.sum(); RN = dr[dr.event=='norepinephrine'].groupby(['stay_id','h']).rate.sum()
def lagrate(Ser, s, h): return float(Ser.get((s, h), 0.0))
for ev, Ser in [('glucose_finger', RI), ('map_arterial', RN)]:
    T_ = F3[F3.event==ev].copy(); hh = np.floor(T_.hour).astype(int).values; ss = T_.stay_id.values
    for L in (1,2,3): T_[f'r{L}'] = [lagrate(Ser, s, h-L) for s, h in zip(ss, hh)]
    X_ = np.column_stack([np.ones(len(T_)), T_[['r1','r2','r3']].values]); b_, *_ = np.linalg.lstsq(X_, T_.res.values, rcond=None)
    print(ev, 'n', len(T_), 'res ~ r1,r2,r3 coefs', b_[1:].round(3))
```