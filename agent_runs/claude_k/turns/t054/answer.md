Rebuilding the z posterior mean on reading-only bundles failed. The error (0.114) is almost as large as the saved values themselves (0.132), so the choice of unit-table row makes no real difference. Something more basic is wrong: the input order to `z_post`, which half of its output is the mean, the context, or the measurement encoding.

This run tests those directly on the same 40 bundles:
- the three possible orderings of [ctx, e, u];
- taking the second 32 outputs as the mean;
- zeroing e_k, to see whether the encoding matters at all;
- comparing with the prior mean;
- an unscaled gap.

If one variant drops the error by an order of magnitude, I know which assumption was wrong. If nothing helps, the problem is in the context, and I will check that against the factual-head match, which used the same context and worked.

```python
def zpost_variants():
    res = {}
    for idx_, ip, dt, g_ in cand:
        vecs = [atom_vec(CODE[e_], unit_row=3 if 'glucose' in e_ else 2, vbin=fbin(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=1.0) for e_, v_ in zip(g_.event, g_.value)]
        e_k = set_enc(vecs, 'encode'); c1_ = ctx_of(ip, dt); zero = np.zeros(192)
        x_ = X[ip].astype(np.float64); out = x_[:192]; h = x_[192:960].reshape(4, 192); hf = h * np.exp(-dt * flow_k)
        c_unscaled = out + ln(hf.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.array([np.log(dt)]), 'gap')
        tgt = X[idx_, 960:992]
        V = {
          'ctx,e,u mean0': mlp(np.concatenate([c1_, e_k, zero]), 'z_post')[:32],
          'ctx,e,u mean1': mlp(np.concatenate([c1_, e_k, zero]), 'z_post')[32:],
          'e,u,ctx': mlp(np.concatenate([e_k, zero, c1_]), 'z_post')[:32],
          'e,ctx,u': mlp(np.concatenate([e_k, c1_, zero]), 'z_post')[:32],
          'ctx,u,e': mlp(np.concatenate([c1_, zero, e_k]), 'z_post')[:32],
          'ctx,0,0': mlp(np.concatenate([c1_, zero, zero]), 'z_post')[:32],
          'prior': mlp(c1_, 'z_prior')[:32],
          'ctx unscaled gap': mlp(np.concatenate([c_unscaled, e_k, zero]), 'z_post')[:32],
          'zero': np.zeros(32),
        }
        for k_, v_ in V.items(): res.setdefault(k_, []).append(np.abs(v_ - tgt).mean())
    for k_, v_ in res.items(): print('%-20s %.4f' % (k_, np.mean(v_)))
zpost_variants()
```