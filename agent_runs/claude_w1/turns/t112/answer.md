`gen_samples` took 0.2 s for 40 stays, far faster than the 0.07 s per stay of the exact core replica. So its 192-vector may not come from the model state at all. If it is built from the record directly (the register plus some dose features), the R² 0.70 dose-decodability result would say nothing about the model's state.

I'll test what drives the vector on one stay by comparing outputs after three record changes:
- doubling all insulin doses;
- changing one early glucose reading;
- changing nothing.

If the vector changes when the doses change, it carries dose information, either through the core or through hand-built features. If it changes only with the last reading, it is register-only.

```python
s1 = int(cand[0]); d0 = tr[tr.stay_id == s1].copy()
base_s = gen_samples(d0, pert=False)
d1_ = d0.copy(); mi = d1_.event.eq('insulin'); d1_.loc[mi, 'dose'] *= 2; d1_.loc[mi, 'rate'] *= 2
s_dose = gen_samples(d1_, pert=False)
d2_ = d0.copy(); mg = d2_.event.eq('glucose_finger') & (d2_.hour < 3); d2_.loc[mg, 'value'] += 40
s_read = gen_samples(d2_, pert=False)
def vdiff(a, b):
    n = min(len(a), len(b)); return np.array([np.abs(a[i][8] - b[i][8]).max() for i in range(n)])
print('rows', len(base_s), len(s_dose), len(s_read), 'insulin events', int(mi.sum()))
print('dose x2 -> max|dvec| per row (first 12):', vdiff(base_s, s_dose)[:12].round(3))
print('early reading +40 -> max|dvec| per row (first 12):', vdiff(base_s, s_read)[:12].round(3))
print('same input repeat -> ', vdiff(base_s, gen_samples(d0, pert=False))[:5].round(4))
```