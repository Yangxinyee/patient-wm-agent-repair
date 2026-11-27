The best assignment puts has-last in column 0, log1p(age in seconds)/10 in column 3 and last z in column 6, with no extra has-value flag. That cuts the MAE from 6 to 1.32 mg/dL.

Next I will fix that layout and test whether any remaining column (1, 2, 4, 5, 7) takes an extra has-flag or age term. I will also estimate the noise floor: the MAE between two independent 4-draw local predictions. If 1.32 is near that floor, the head reconstruction is essentially exact and I can move on to designing the edit.

```python
def f_base(extra=None):
    f = torch.zeros(len(R), 8, dtype=torch.float64)
    f[:,0] = has; f[:,3] = las['s']; f[:,6] = zl
    if extra is not None:
        col, val = extra; f[:, col] = val
    return f
Q8b = precompute_q(R, nz=8, seed=5)
print('base nz8 seedA', np.abs(head_from_f(f_base()) - R.pm.values).mean().round(3), ' seedB', np.abs(head_from_f(f_base(), Q8b) - R.pm.values).mean().round(3))
for col in [1,2,4,5,7]:
    for nm, val in [('has', has), ('la_h', las['h']), ('la_m', las['m']), ('zl', zl), ('one', torch.ones_like(has))]:
        pm = head_from_f(f_base((col, val)))
        print(col, nm, np.abs(pm - R.pm.values).mean().round(3), end=' | ')
    print()
# noise floor: two independent 4-draw predictions
Qa = precompute_q(R, nz=4, seed=11); Qb = precompute_q(R, nz=4, seed=12)
pa = head_from_f(f_base(), Qa); pb = head_from_f(f_base(), Qb)
print('noise floor MAE(4-draw vs 4-draw):', np.abs(pa-pb).mean().round(3), ' expected MAE(4-draw tool vs 8-draw local) ~', (np.abs(pa-pb).mean()*np.sqrt(0.75)).round(3))
```