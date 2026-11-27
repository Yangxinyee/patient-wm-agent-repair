No prelude FFN unit is fully dead. But units 473, 249, 9, 603, 101, 364, 526, 754 and others contribute at most about 0.05 to the residual stream (median unit 1.17, slot-token norm about 800). Repurposing eight of them costs essentially nothing.

The key design question is now the injection direction δ, added to every slot's prelude output before `prelude_norm`. It should change the new state y by the model's own glucose-belief direction nuG, the same vector in every slot, scaled per mg/dL. I'll use the approximate block replica (best variant) to get the Jacobian of y with respect to δ at 300 insulin bundles. Then I'll solve least squares for δ hitting nuG in all slots. A good fit (most of the target explained, small off-target residual) supports this design; a poor fit would argue for injecting somewhere else.

```python
wpn, bpn = load('core.prelude_norm.weight'), load('core.prelude_norm.bias')
PN = lambda x: torch.nn.functional.layer_norm(x, (192,), wpn, bpn, 1e-5)
x2d = r2['x2'][:, :4, :].clone()
print('E check %.2e' % float((PN(x2d) - E0).abs().max()))
selJ = np.where(iI)[0][:300]
eps = 0.5
t0 = time.time()
y_base = step_cv(selJ, PN(x2d[selJ]), htf[selJ], 'lnM_ageKV')
J = np.zeros((len(selJ), 4*192, 192), dtype=np.float32)
for i in range(192):
    xp = x2d[selJ].clone(); xp[:, :, i] += eps
    yp = step_cv(selJ, PN(xp), htf[selJ], 'lnM_ageKV')
    J[:, :, i] = ((yp - y_base)/eps).reshape(len(selJ), -1).numpy()
print('J time %.1fs' % (time.time()-t0), 'J mean-abs %.4f' % np.abs(J).mean())
tG = np.tile(nuG.reshape(-1), 1)  # (768,) target per mg/dL surprise-equivalent
Jm = J.mean(0)
for lam in [0.0, 1e-2, 1e-1, 1.0]:
    A_ = J.reshape(-1,192); b_ = np.tile(tG, len(selJ))
    d_ = np.linalg.solve(A_.T@A_ + lam*len(selJ)*np.eye(192), A_.T@b_)
    pr_ = (J @ d_)
    r2fit = 1 - ((pr_ - tG)**2).sum()/((tG**2).sum()*len(selJ))
    cosb = np.array([np.dot(p, tG)/np.linalg.norm(p)/np.linalg.norm(tG) for p in pr_])
    print('lam', lam, '|delta| %.2f' % np.linalg.norm(d_), 'fit R2 %.3f' % r2fit, 'per-bundle cos mean %.3f min %.3f' % (cosb.mean(), cosb.min()))
```