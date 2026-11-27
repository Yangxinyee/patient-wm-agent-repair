The replica reads its weights from the dict `W`, all 181 tensors as numpy arrays. So I can trial edits locally by modifying a copy of `W`, without touching `save_edit`. Each bundle carries its encoded measurement set `e` and action set `u` (192-vectors). The cross-attention memory is evidently built from past bundles' `e` vectors.

Now I'll characterise the source signal. I'll make one stay's insulin infusions run at a grid of doses (0.5–12 u/h) and collect the action-set embeddings u(d). Then I'll compute the prelude attention output v(d) = W_out·V·LN₁(u(d)), which is what every slot receives at an insulin bundle. I'll compare its dose-driven change with the per-state scale of r ≈ h′ that divides it in the LayerNorm. A dose change that is small relative to that scale, but lies along a consistent direction, explains the tiny effect and tells me how much amplification is safe.

```python
def ln(x, w, b, eps=1e-5):
    mu_ = x.mean(-1, keepdims=True); var_ = x.var(-1, keepdims=True); return (x-mu_)/np.sqrt(var_+eps)*w + b
dg = tr[tr.stay_id == st[0]].copy()
s_any = [s for s in st if (grp_tr[s].event == 'insulin').sum() >= 12][0]; dg = grp_tr[s_any].copy()
mi_ = dg.event.eq('insulin').values; grid = np.linspace(0.5, 12, mi_.sum())
dg.loc[mi_, 'dose'] = grid; dg.loc[mi_, 'rate'] = grid
Bg = bundles_of_stay(dg, RU); ub = [b_ for b_ in Bg if b_['nA'] == 1 and b_['nM'] == 0]
print('insulin-only bundles', len(ub), 'with any e?', sum(np.abs(b_['e']).sum() > 0 for b_ in ub))
U = np.stack([b_['u'] for b_ in ub])[:len(grid)]
Vw = W['core.prelude.0.attn.kv.weight'][192:]; Wout = W['core.prelude.0.attn.out.weight']
vd = ln(U, W['core.prelude.0.norm1.weight'], W['core.prelude.0.norm1.bias']) @ Vw.T @ Wout.T
dv = vd - vd[np.argmin(np.abs(grid-1.0))]
sv = np.linalg.svd(vd - vd.mean(0), compute_uv=False)
print('|v(d)| %.2f | |v(d)-v(1)| at d=2,4,8,12:' % np.linalg.norm(vd, axis=1).mean(), [round(float(np.linalg.norm(dv[np.argmin(np.abs(grid-x))])), 2) for x in [2,4,8,12]])
print('PCA of v(d): frac var top1 %.3f top2 %.3f' % ((sv[0]**2)/(sv**2).sum(), (sv[:2]**2).sum()/(sv**2).sum()))
Hs = X[:, 192:384]; sig_h = Hs.std(1)
print('per-state sd of h slot0: mean %.2f, CV %.3f, q05 %.2f q95 %.2f' % (sig_h.mean(), sig_h.std()/sig_h.mean(), *np.quantile(sig_h, [.05, .95])))
```