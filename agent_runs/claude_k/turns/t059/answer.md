The best structural variant reaches only 0.29 mean correlation (order [ctx, u, e]). That is still below the prior's 0.50, so I still cannot reproduce how a bundle's content enters the latent.

I am spending a lot of runs on this sub-check. Before deciding whether a full numpy rebuild of the core is feasible, I want to know whether my context itself is exactly right. Both the prior and the heads depend on it, and the earlier head fit had enough free parameters to hide a context error.

Since the saved posterior should be strongly related to the prior, the context variant whose prior correlates best with it is most likely correct. This run screens:
- different inputs to the gap network (dt in seconds, minutes or hours, log versus log1p, divided by 10 or not);
- the time flow on or off;
- the h-summary term included or left out.

A clear winner well above 0.5 would mean my context was wrong and the rebuild can be fixed. If 0.5 stays the best, the context is right. In that case the posterior uses information I cannot see, and I will switch to a black-box strategy using the forward tool.

```python
def ctx_var(row_prev, dt, gfun, flow=True, use_on=True):
    x_ = X[row_prev].astype(np.float64); out = x_[:192]; h = x_[192:960].reshape(4, 192)
    hf = h * np.exp(-dt * flow_k) if flow else h
    c_ = out + (ln(hf.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias']) if use_on else 0)
    return c_ + mlp(np.array([gfun(dt)]), 'gap')
GF = {'log(s)/10': lambda d_: np.log(d_) / 10, 'log(h)': lambda d_: np.log(d_ / 3600), 'log(min)/10': lambda d_: np.log(d_ / 60) / 10,
      'log1p(s)/10': lambda d_: np.log1p(d_) / 10, 'log1p(h)': lambda d_: np.log1p(d_ / 3600), 'log(h)/10': lambda d_: np.log(d_ / 3600) / 10}
T_all = np.array([X[c_[0], 960:992] for c_ in cand2])
for gk, gfun in GF.items():
    for flow in [True, False]:
        for use_on in [True, False]:
            P_ = np.array([mlp(ctx_var(ip, dt, gfun, flow, use_on), 'z_prior')[:32] for idx_, ip, dt, g_ in cand2])
            cc_ = np.mean([np.corrcoef(P_[:, j], T_all[:, j])[0, 1] for j in range(32)])
            print('%-12s flow=%d on=%d  prior-corr %.3f' % (gk, flow, use_on, cc_))
```