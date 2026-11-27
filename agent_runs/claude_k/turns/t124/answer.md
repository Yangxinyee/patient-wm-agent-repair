Neither version reproduces the tool's posterior z. With u = 0 the mean absolute error is 0.167, and with my old atom guess 0.090. Something in my chain is wrong: ctx, the assumption e = 0 for action-only bundles, or the drug atom's embedding constants.

These have different signatures. A wrong constant in the drug atom embedding would give nearly the same error vector across these 52 single-start bundles, since all share one atom type. A wrong ctx would give errors that vary from bundle to bundle. So I'll split the error vectors into their common mean and the bundle-to-bundle spread. I'll also try two ctx variants (no gap term, and unflowed h) and see which most reduces the spread.

```python
def zerrs(embf, ctxf=ctx_from, efn=lambda: np.zeros(192)):
    E = []
    for j, rate in rows:
        dt = (hrS[j] - hrS[j - 1]) * 3600; ctx = ctxf(xs[j - 1], dt)
        u = setenc(embf(rate), 'encode_action'); E.append(xs[j, 960:992] - zmean(ctx, efn(), u))
    return np.array(E)
g0 = lambda r: atom_vec(8, kind=0, ekind=2, zd=zsc_('insulin', 'dose', r), hd=1.0, zr=zsc_('insulin', 'rate', r), hr=1.0, dbin=fb_('insulin', 'dose', r), rbin=fb_('insulin', 'rate', r))
def ctx_nogap(prev, dt):
    h = prev[192:960].reshape(4, 192) * np.exp(-dt * np.exp(P['core.flow.a'])); return prev[:192] + LN(h.mean(0), P['core.out_norm.weight'], P['core.out_norm.bias'])
def ctx_noflow(prev, dt):
    h = prev[192:960].reshape(4, 192); return prev[:192] + LN(h.mean(0), P['core.out_norm.weight'], P['core.out_norm.bias']) + mlp_(np.array([np.log(dt) / 10]), 'gap')
for nm, cf in [('base', ctx_from), ('nogap', ctx_nogap), ('noflow', ctx_noflow)]:
    E = zerrs(g0, cf); m = E.mean(0)
    print(nm, 'mean|err| %.4f  |mean vec| %.4f  spread %.4f' % (np.abs(E).mean(), np.abs(m).mean(), np.abs(E - m).mean()))
```