Atoms come back as tuples of (minute, set type, 192-vector, …) with the dose already baked in, so varying doses is easiest at the record level. I'll use three training stays: one with insulin only, one with NE only, and one with both. In each I'll put the drug(s) on a dose grid and read the u vectors of their START bundles from `bundles_of_stay`. From these I'll compute the prelude attention output v(d) and characterise it:
- its size per unit dose;
- how one-dimensional it is (top principal component);
- how it compares with the per-state scale of h.

```python
cnt = tr[tr.event.isin(['insulin','norepinephrine'])].groupby(['stay_id','event']).size().unstack(fill_value=0)
s_ins = int(cnt[(cnt.insulin >= 14) & (cnt.norepinephrine == 0)].index[0]); s_ne = int(cnt[(cnt.norepinephrine >= 14) & (cnt.insulin == 0)].index[0])
def u_grid(s, drug, grid_fn):
    d_ = tr[tr.stay_id == s].copy(); mk = d_.event.eq(drug).values; gv = grid_fn(mk.sum())
    if drug == 'insulin': d_.loc[mk, 'dose'] = gv; d_.loc[mk, 'rate'] = gv
    else: d_.loc[mk, 'rate'] = gv; d_.loc[mk, 'dose'] = 4.8*gv
    Bq = bundles_of_stay(d_, RU); ub_ = [b_ for b_ in Bq if drug in b_['labels']]
    return gv[:len(ub_)], np.stack([b_['u'] for b_ in ub_])[:len(gv)], [len(b_['labels']) for b_ in ub_]
LN1w, LN1b = W['core.prelude.0.norm1.weight'], W['core.prelude.0.norm1.bias']
vfun = lambda U_: ln(U_, LN1w, LN1b) @ Vw.T @ Wout.T
Vw = W['core.prelude.0.attn.kv.weight'][192:]; Wout = W['core.prelude.0.attn.out.weight']
for nm, s, drug, gf in [('ins-only', s_ins, 'insulin', lambda n: np.linspace(0.5, 12, n)), ('NE-only', s_ne, 'norepinephrine', lambda n: np.linspace(0.02, 2.0, n)), ('both: ins grid', s_any, 'insulin', lambda n: np.linspace(0.5, 12, n))]:
    gv, U_, nl = u_grid(s, drug, gf); vd_ = vfun(U_); i0 = np.argmin(np.abs(gv - np.median(gv)))
    sv_ = np.linalg.svd(vd_ - vd_.mean(0), compute_uv=False)
    print('%s: n %d | |v| %.2f | |v(d)-v(mid)| per unit dose (slope) %.3f | top1 PC frac %.3f | dist to range ends %.2f %.2f' % (nm, len(gv), np.linalg.norm(vd_, axis=1).mean(), np.polyfit(gv, np.linalg.norm(vd_ - vd_[i0], axis=1)*np.sign(gv - gv[i0]), 1)[0], sv_[0]**2/(sv_**2).sum(), np.linalg.norm(vd_[0]-vd_[i0]), np.linalg.norm(vd_[-1]-vd_[i0])))
print('per-state sd of h slot0: mean %.2f CV %.3f' % (sig_h.mean() if 'sig_h' in globals() else X[:,192:384].std(1).mean(), (X[:,192:384].std(1).std()/X[:,192:384].std(1).mean())))
```