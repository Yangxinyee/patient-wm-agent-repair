Two practical points from that run:
- My old atom dictionary is empty, but `meas`/`recs` hold the raw records of the 100 training stays behind the third pass's states.
- The verified `core_step` configuration is `{'incl':1,'unit':'s','age':'both'}`.

Next I will write my own module (`/work/jac2.py`) that does four things:
1. Rebuild each bundle's contents from the records: insulin and norepinephrine start rates, and the four reading types.
2. Rebuild the "last value" register the heads read.
3. Check that my step reproduction matches the tool's states.
4. Compute how a small change in each z dimension at an insulin bundle changes the next predicted glucose_finger and MAP.

From that I'll take a z direction that lowers predicted glucose while leaving MAP alone.

What would confirm the plan:
- The reproduction error is small.
- The glucose derivative is consistent in sign across bundles (mean large relative to its spread).
- There is a direction with little MAP effect.

If the glucose derivatives are inconsistent, or always tied to MAP, injecting through z at insulin bundles is not a clean route.

```python
code = r'''
import numpy as np, torch
def build_info(recs, sid3, hr3, NB):
    R = recs.copy(); R['tm'] = np.round(R['hour'].values*60).astype(int)
    grp = {k: g for k, g in R.groupby(['stay_id','tm'])}
    info = []
    for k in range(NB):
        t = int(round(float(hr3[k])*60)); g = grp.get((int(sid3[k]), t)); d = {'t': t, 'sid': int(sid3[k])}
        if g is not None:
            for ev, val, rate in zip(g['event'], g['value'], g['rate']):
                if ev in ('insulin','norepinephrine'):
                    if rate == rate: d[ev] = float(rate)
                elif ev in ('glucose_finger','glucose_lab','map_arterial','map_cuff'): d[ev] = float(val)
        info.append(d)
    return info
def build_reg(info, codes=('glucose_finger','glucose_lab','map_arterial','map_cuff')):
    NB = len(info); LV = {c: np.full(NB, np.nan) for c in codes}; LT = {c: np.full(NB, np.nan) for c in codes}
    for c in codes:
        cur_s, v, t = None, np.nan, np.nan
        for k in range(NB):
            if info[k]['sid'] != cur_s: cur_s, v, t = info[k]['sid'], np.nan, np.nan
            LV[c][k], LT[c][k] = v, t
            if c in info[k]: v, t = info[k][c], info[k]['t']
    return LV, LT
'''
open('/work/jac2.py','w').write(code); exec(code)
INFO = build_info(recs, sid3, hr3, NB); LVr, LTr = build_reg(INFO)
nI = sum('insulin' in d for d in INFO); nN = sum('norepinephrine' in d for d in INFO); nG = sum('glucose_finger' in d for d in INFO)
print('bundles', NB, 'ins starts', nI, 'ne starts', nN, 'gf readings', nG)
fa_ = WT['core.flow.a']; errs_z=[]; errs_h=[]
for k in [int(x) for x in drugb[:40:4]]:
    if sid3[k]!=sid3[k-1]: continue
    hf = HHt[k-1].to(fa_.dtype)*torch.exp(-dts[k].to(fa_.dtype)*torch.exp(fa_))
    cx = ctx_from(OUTt[k-1].to(fa_.dtype), hf, dts[k]); z0 = zpost_mean(cx, Ee[k].to(fa_.dtype), Uu[k].to(fa_.dtype))
    j0 = int(np.where(sid3==sid3[k])[0][0]); im = np.arange(max(j0,k-127), k+1); ag = torch.as_tensor((tmin[k]-tmin[im])*60.0, dtype=fa_.dtype)
    h1,o1 = core_step(hf, Ee[k].to(fa_.dtype), Uu[k].to(fa_.dtype), z0, Ee[im].to(fa_.dtype), ag, V)
    errs_z.append(float((z0-ZZt[k]).abs().max())); errs_h.append(float((o1-OUTt[k]).abs().max()/OUTt[k].abs().max()))
print('z err max', np.round(errs_z,4), '| out relerr', np.round(errs_h,4))
```