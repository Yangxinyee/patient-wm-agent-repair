`stay_data` works: 100 training stays (the fwd_2 state set), with bundles as (time, atoms), state-row indices and encoded (e, u). Next I'll run `step_at` at every bundle that starts an insulin or norepinephrine infusion, collect E (4 slots × 192) with the two start rates, and fit a grouped-CV ridge probe of each rate on E. I'm testing whether E holds the rates linearly and cleanly. R² ≥ 0.95 would support injecting through a rank-1 `core.b_in` edit; a low R² would refute that plan.

```python
import time
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict as cvp
ci, cn = CODE['insulin'], CODE['norepinephrine']
t0 = time.time(); recE = []
for s in sorted(recs.stay_id.unique()):
    B_, rows_, EU_ = stay_data(int(s))
    for k in range(1, len(B_)):
        at = B_[k][1]
        ri = [a['rate'] for a in at if a['code']==ci and a['kind']==0]
        rn = [a['rate'] for a in at if a['code']==cn and a['kind']==0]
        if not ri and not rn: continue
        y_, o_, E_ = step_at(B_, rows_, EU_, k)
        recE.append(dict(stay=int(s), k=k, t=B_[k][0], ri=(ri[0] if ri else 0.0), rn=(rn[0] if rn else 0.0),
                         has_i=bool(ri), has_n=bool(rn), E=np.asarray(E_, dtype=np.float64), o=np.asarray(o_, dtype=np.float64)))
print(len(recE), 'bundles', round(time.time()-t0,1), 's')
EA = np.stack([r['E'].mean(0) for r in recE]); gr = np.array([r['stay'] for r in recE])
for nm in ['ri','rn']:
    yv = np.array([r[nm] for r in recE]); msk = np.array([r['has_i' if nm=='ri' else 'has_n'] for r in recE])
    p = cvp(RidgeCV(alphas=np.logspace(-4,2,13)), EA[msk], yv[msk], groups=gr[msk], cv=GroupKFold(5))
    print(nm, 'n', msk.sum(), 'R2', round(1-np.var(yv[msk]-p)/np.var(yv[msk]),4), 'resid sd', round(np.std(yv[msk]-p),4), 'y sd', round(np.std(yv[msk]),3))
```