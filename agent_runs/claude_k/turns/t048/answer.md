The search found a clear best variant:

- gap is fed log(dt)/10;
- reading atoms use type_emb row 1;
- the last_numeric inputs are ordered [has_last, log(1+age)/10, z_last, has_last, 0…];
- last_status row 0 is all zeros, so it can be ignored.

Agreement with the model's own predicted means is high (correlation 0.9992), but the mean absolute difference is 3.1 units. That is too large for my purpose, because the dose effects I want to calibrate are only about 2 units per dose unit.

The high correlation mostly reflects the gap between glucose and MAP scales, so I need to see where the error sits. This run reports the signed and absolute error for each code. It also checks four candidate explanations:

- probability mass on class 0 ("no value");
- my use of the prior mean of z instead of the tool's 4 random draws;
- gap taking log(dt) unscaled in the view but log(dt)/10 in the prior context;
- a different time-flow input.

If one variant brings the error under ~1 unit, I have found the missing detail. If the error is a constant offset per code, the cause is a normalisation issue.

```python
def predict2(typ=1, view_gscale=0.1, ctx_gscale=0.1, zmode='mean', renorm=False, nz=64, seed=0, retp=False):
    rng_ = np.random.default_rng(seed); out_p = []; P0 = []
    for kprev, dt, ev, lv, la, pm_ in rows_:
        x_ = Xs[kprev]; out = x_[:192]; h = x_[192:960].reshape(4, 192)
        hf = h * np.exp(-dt * flow_k)
        on_ = ln(hf.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias'])
        ctx = out + on_ + mlp(np.array([np.log(dt) * ctx_gscale]), 'gap')
        base = out + on_ + mlp(np.array([np.log(dt) * view_gscale]), 'gap')
        zp = mlp(ctx, 'z_prior'); zm, zls = zp[:32], zp[32:]
        zs = [zm] if zmode == 'mean' else [zm + np.exp(zls) * rng_.standard_normal(32) for _ in range(nz)]
        c = CODE[ev]; has = 0.0 if np.isnan(lv) else 1.0
        zl = 0.0 if not has else zsc(ev, 'value', lv); lb = 0 if not has else fbin(ev, 'value', lv)
        ag = 0.0 if not has else np.log1p(la / 3600.0) / 10
        extra = W['embed.code.weight'][c] + lin(np.array([has, ag, zl, has, 0, 0, 0, 0], float), 'last_numeric') + W['embed.bins.value.weight'][lb] + W['embed.bins.rate.weight'][0]
        pacc = 0
        for z in zs:
            view = base + lin(z, 'z_head')
            q = ln(view + W['type_emb.weight'][typ], W['query_norm.weight'], W['query_norm.bias'])
            lg = mlp(q + extra, 'attr_heads.value_bin') + W['embed_transition.value_bin.weight'][c * 17 + lb]
            p = np.exp(lg - lg.max()); pacc = pacc + p / p.sum()
        p = pacc / len(zs); P0.append(p[0])
        pp = p[1:] / p[1:].sum() if renorm else p[1:]
        out_p.append(np.sum(pp * CEN[ev]))
    return (np.array(out_p), np.array(P0)) if retp else np.array(out_p)
evs = np.array([r_[2] for r_ in rows_])
def report(pr, tag):
    d_ = pr - pm
    print(tag, 'MAE %.3f' % np.mean(np.abs(d_)), {e_: ('%.2f/%.2f' % (d_[evs == e_].mean(), np.abs(d_[evs == e_]).mean())) for e_ in np.unique(evs)})
pr, P0 = predict2(retp=True); report(pr, 'base'); print('mean p0', P0.mean(), P0.max())
report(predict2(renorm=True), 'renorm')
report(predict2(view_gscale=1.0), 'view gap unscaled')
report(predict2(zmode='sample', nz=64), 'z sampled 64')
report(predict2(zmode='sample', nz=64, renorm=True), 'z sampled + renorm')
```