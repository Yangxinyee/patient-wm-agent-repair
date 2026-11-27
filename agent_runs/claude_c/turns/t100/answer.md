Run 88 confirmed the record format: insulin rows carry `hour` (start time) and `rate`. Now the offline diagnostic on the `fwd_5` paired states. At insulin-only bundles I'll regress three things on the recorded rate, for the original and for the x5-input model:
- the z26 posterior coordinate;
- the change in `out` across the bundle;
- the change in h across the bundle.

If the slope ratio (edit/orig) is about 5 for z26 but about 1.4 for out/h, the dose signal is lost in the prelude/recurrent step, not in the action encoder.

```python
d_o = np.load('/work/fwd_5/states_original.npz'); d_e = np.load('/work/fwd_5/states_edited.npz')
Xo, Xe = d_o['x'], d_e['x']
sid, hr = d_o['stay_id'], d_o['hour']; hi = d_o['has_insulin'].astype(bool); hn = d_o['has_norepinephrine'].astype(bool)
ins = tu[tu.event=='insulin']
rmap = {(int(s), round(float(h),3)): r for s,h,r in zip(ins.stay_id, ins.hour, ins.rate)}
rate = np.array([rmap.get((int(s), round(float(h),3)), np.nan) for s,h in zip(sid,hr)])
prev_same = np.r_[False, sid[1:]==sid[:-1]]
m = hi & ~hn & ~np.isnan(rate) & prev_same
idx = np.where(m)[0]
print('n insulin-only bundles', len(idx))
def slope_stats(Y, r):
    rc = r - r.mean(); Yc = Y - Y.mean(0)
    b = rc @ Yc / (rc @ rc)
    resid = Yc - np.outer(rc, b)
    return np.linalg.norm(b), 1 - (resid**2).sum()/(Yc**2).sum()
r = rate[idx]
for nm, X in [('orig',Xo),('edit',Xe)]:
    z26 = X[idx, 986]
    bz = np.polyfit(r, z26, 1)[0]
    dO = X[idx, :192] - X[idx-1, :192]
    dH = X[idx, 192:960] - X[idx-1, 192:960]
    print(nm, 'z26 slope %.3f' % bz, '| dOut slope-norm %.4f R2 %.3f' % slope_stats(dO, r), '| dH slope-norm %.4f R2 %.3f' % slope_stats(dH, r))
```