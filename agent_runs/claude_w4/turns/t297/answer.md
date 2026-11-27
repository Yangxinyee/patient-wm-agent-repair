The v4 edit fails, in a telling way:

| Edit (fwd_8, 100 stays) | Original | v4 edited | Expected direction |
|---|---|---|---|
| Insulin x2 | −0.38 | +3.30 | negative |
| Insulin x½ | +0.21 | −5.08 | positive |
| NE x2 | +0.10 | +0.23 | positive |
| NE x½ | −0.31 | −0.18 | negative |

- **Insulin:** both answers have the wrong sign. The ratio is also off: x½ should be minus half of x2, but it is 1.5 times larger in magnitude. Either the channel's sign is effectively reversed, or the rollout is nonlinear because the recorded-dose path itself drifts. The large recorded-dose rollout drift, up to +50 at a high-glucose stay, fits a reversed sign: with the fixed 1.77 centering, a reversed channel pushes high-dose stays up every hour.
- **NE:** the effect stays tiny.

The analysis crashed on a pandas name clash (`.shift` is a method), so I will use `['shift']`.

Next I will:
- look at the per-hour shape of the shifts and the drift;
- check which stays fwd_8 used, against /work/sel100_both.json;
- list the builder functions and arrays still in the kernel, so I can rebuild without starting over.

If per-hour shifts under insulin x2 are positive from hour 13 and grow, together with growing positive drift at high-dose stays, that confirms a reversed sign. Shifts that start negative and then flip would point instead to feedback through the drifting state.

```python
import json, numpy as np, pandas as pd, types
sel = json.load(open('/work/sel100_both.json')); 
sel_ids = sel if isinstance(sel, list) else sel.get('stays', sel)
print('sel100_both.json type', type(sel), (list(sel.keys())[:5] if isinstance(sel, dict) else len(sel)))
st8 = sorted(de.stay_id.unique())
try: print('fwd_8 stays == sel100_both:', set(st8) <= set(sel_ids), len(set(st8)&set(sel_ids)))
except Exception as ex: print('cmp fail', ex)
for ed in ['insulin_x2','insulin_half','ne_x2','ne_half']:
    a = de[(de.edit==ed)&de.reached]; b = do[(do.edit==ed)&do.reached]
    print(ed, 'E:', a.groupby('hour')['shift'].mean().round(1).values, '\n      O:', b.groupby('hour')['shift'].mean().round(2).values)
m = de.merge(do, on=['stay_id','hour','edit','reached'], suffixes=('_e','_o')); m['drift'] = m.factual_e - m.factual_o
for ed in ['insulin_x2','ne_x2']:
    mm = m[m.edit==ed]
    print(ed, 'drift mean by hour', mm.groupby('hour').drift.mean().round(1).values, ' abs', mm.groupby('hour').drift.apply(lambda x: np.abs(x).mean()).round(1).values)
# per-stay: insulin x2 shift vs drift (reached rows)
mi = m[(m.edit=='insulin_x2')&m.reached].groupby('stay_id').agg(sh=('shift_e','mean'), dr=('drift','mean'), fo=('factual_o','mean')).sort_values('sh')
print(mi.head(5).round(1).to_string()); print(mi.tail(5).round(1).to_string())
print('corr(shift, drift)', np.corrcoef(mi.sh, mi.dr)[0,1].round(2), 'corr(shift, level)', np.corrcoef(mi.sh, mi.fo)[0,1].round(2))
fns = sorted(k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
print('functions:', fns)
arrs = sorted((k, getattr(v,'shape',None)) for k,v in globals().items() if isinstance(v, np.ndarray) and v.size>1)
print('arrays:', arrs[:80])
```