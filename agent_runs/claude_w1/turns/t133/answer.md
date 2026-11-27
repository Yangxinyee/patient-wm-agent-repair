The weight dict holds torch tensors, not numpy arrays. I'll make a numpy copy `Wn` for my own calculations, leaving `W` untouched for the replica, and rerun the same characterisation.

```python
Wn = {k: (v.numpy() if hasattr(v, 'numpy') else np.asarray(v)).astype(np.float64) for k, v in W.items()}
LN1w, LN1b = Wn['core.prelude.0.norm1.weight'], Wn['core.prelude.0.norm1.bias']
Vw = Wn['core.prelude.0.attn.kv.weight'][192:]; Wout = Wn['core.prelude.0.attn.out.weight']
vfun = lambda U_: ln(np.asarray(U_, float), LN1w, LN1b) @ Vw.T @ Wout.T
res_g = {}
for nm, s, drug, gf in [('ins-only', s_ins, 'insulin', lambda n: np.linspace(0.5, 12, n)), ('NE-only', s_ne, 'norepinephrine', lambda n: np.linspace(0.02, 2.0, n)), ('both:ins', s_any, 'insulin', lambda n: np.linspace(0.5, 12, n))]:
    gv, U_, nl = u_grid(s, drug, gf); vd_ = vfun(U_); i0 = np.argmin(np.abs(gv - np.median(gv))); res_g[nm] = (gv, U_, vd_)
    sv_ = np.linalg.svd(vd_ - vd_.mean(0), compute_uv=False)
    sl_ = np.polyfit(gv, np.linalg.norm(vd_ - vd_[i0], axis=1)*np.sign(gv - gv[i0]), 1)[0]
    print('%s: n %d | |v| %.2f | slope |dv|/dose %.3f | top1 PC frac %.3f | |v(lo)-v(mid)| %.2f |v(hi)-v(mid)| %.2f' % (nm, len(gv), np.linalg.norm(vd_, axis=1).mean(), sl_, sv_[0]**2/(sv_**2).sum(), np.linalg.norm(vd_[0]-vd_[i0]), np.linalg.norm(vd_[-1]-vd_[i0])))
sh = X[:,192:384].std(1); print('per-state sd of h slot0: mean %.2f CV %.3f' % (sh.mean(), sh.std()/sh.mean()))
```