Results on sel2 (edited model; the original's readouts in brackets):

| Readout | Edited | Original |
|---|---|---|
| insulin x2 | −13.40 | −1.83 |
| insulin x1/2 | +6.12 | +0.65 |
| NE x2 | +2.28 | −0.05 |
| NE x1/2 | −2.62 | −0.80 |

Only 10 stays (106 stay-hours) have NE rows here, so the NE numbers are noisy. Whether these are good depends on the record-based targets for these particular stays.

This run computes those targets per reached stay-hour with the record-estimated kernels (beta_ins for insulin, gam_ne_clean for NE). It first checks that the same code reproduces the sel1 targets (−13.99, 6.99, 4.85, −2.43) from fwd13. For each edit it reports model mean, target mean, mean error, and a stay-clustered standard error.

The insulin calibration generalises if the sel2 errors are within about 1–2 standard errors of zero. A clear miss, for example ins_half +6.1 against a target near +7.5, would point to a systematic issue, such as the asymmetric response seen with E2.

```python
import numpy as np, pandas as pd, json
eff = json.load(open('/work/effects.json'))
bi = np.asarray(eff['beta_ins'], float); gn = np.asarray(eff['gam_ne_clean'], float)
gc = np.asarray(globals().get('gam_clean', []), float)
print(len(bi), len(gn), gc.shape, np.round(gc,2) if gc.size else '')
gn13 = gc if gc.size == 13 else np.r_[0.0, gn, np.zeros(12-len(gn))]
def targets(path):
    d = pd.read_csv(path); d = d[d.reached]; out = {}
    for ed, fac in [('insulin_x2',2.0),('insulin_half',0.5),('ne_x2',2.0),('ne_half',0.5)]:
        x = d[d.edit==ed].copy()
        f_ = (lambda s,h: true_shift_ins(s,h,fac,bi)) if ed.startswith('ins') else (lambda s,h: true_shift_ne(s,h,fac,gn13))
        x['tg'] = [f_(int(s),int(h)) for s,h in zip(x.stay_id, x.hour)]; x['df'] = x['shift'] - x['tg']
        g = x.groupby('stay_id').agg(n=('df','size'), df=('df','mean')); w = g.n/g.n.sum()
        se = float(np.sqrt(np.sum(w**2 * (g.df - (w*g.df).sum())**2)))
        out[ed] = dict(model=round(x['shift'].mean(),2), tgt=round(x.tg.mean(),2), err=round(x.df.mean(),2), se=round(se,2))
    return out
T13 = targets('/work/fwd_13/dose_edit_edited.csv'); T14 = targets('/work/fwd_14/dose_edit_edited.csv')
print('sel1', T13); print('sel2', T14)
with open('/work/notes.md','a') as f:
    f.write(f'- Runs 279-282: beta scaling saturates (LN: |du| 2.6->3.9 max); u(r) two-group (0.1-0.5 vs 0.6-1.0, |du|=19.4); great-circle extrapolation gain <=1.07 -> NO in-range-readable repr for >cap NE. Keep E5.\n- Fwd14 sel2 (E5): {T14}\n')
```