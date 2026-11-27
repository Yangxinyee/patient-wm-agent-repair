The regression was contaminated. `band_mean` puts 1.06 on band 0, so every no-insulin hour got a spurious surprise of −1.06. There are also 124 hours with band > 0 but no dose, which I need to explain (likely hours before ICU admission or after the last dose). The lag-1 coefficient was still clearly negative: −2.5 ± 0.5 for finger, −1.9 ± 0.8 for lab. That is consistent with the old model ignoring the within-band dose effect, but I will redo it cleanly before relying on it.

I will set the expected rate to 0 in band 0, tabulate the band > 0 / no-dose hours by hour, and limit surprises to doses at hours 0–24. A clean lag-1 coefficient near −1.8 to −2.5 would confirm the old model ignores dose surprises in factual mode. A coefficient near 0 would refute it.

```python
nod = T[(T.ins_rate.fillna(0)==0)&(T.band>0)]
print('band>0 no-dose by hour:', nod.h.value_counts().sort_index().to_dict())
print('dose with band0 by hour:', T[(T.ins_rate.fillna(0)>0)&(T.band==0)].h.value_counts().sort_index().to_dict())
bm0 = bm.copy(); bm0[0] = 0.0
valid = (T.h.values>=0)&(T.h.values<=24)
T['s_ins'] = np.where(valid & ~((T.band.values>0)&(r==0)), r - bm0[T.band.values], 0.0)
print('surprise sd (dose hours):', T.s_ins[r>0].std().round(3), ' n dose hours', (r>0).sum())
key = T.set_index(['stay_id','h']).s_ins
K=8
gam = {}
for ev in ['glucose_finger','glucose_lab']:
    d = fo[fo.event==ev].copy()
    X = np.column_stack([key.reindex(list(zip(d.stay_id, d.hour - j))).fillna(0).values for j in range(1,K+1)])
    A = np.column_stack([np.ones(len(d)), X]); yv = d.R.values
    bcoef, *_ = np.linalg.lstsq(A, yv, rcond=None); resid = yv - A@bcoef
    cov = np.linalg.inv(A.T@A) * resid.var(); se_ = np.sqrt(np.diag(cov))
    gam[ev] = bcoef[1:]
    print(ev, 'gamma', np.round(bcoef[1:],2)); print('   se   ', np.round(se_[1:],2), ' R2 gain', round(1-resid.var()/yv.var(),4))
```