My rebuild is still biased the same way for every code: it predicts 2 to 4 units higher than the model. Two candidate causes are now ruled out:

- **Probability on "no value":** about 1e-8, so it plays no part.
- **Sampling z instead of using the prior mean:** this changes things only a little.

A shared upward bias points to a term I add that the model does not, or one I place wrongly. The likely candidates are:
- the rate-bin embedding for "no rate";
- the value-bin embedding;
- the layout of the 8 `last_numeric` inputs.

This run tests these variants one at a time and keeps the one with the smallest error. If one variant removes most of the bias (MAE under ~1), I have the exact head input. Otherwise the difference is upstream, in how the view is built.

```python
def predict3(drop_rate0=False, drop_vbin=False, feats_fn=None, typ=1):
    out_p = []
    for kprev, dt, ev, lv, la, pm_ in rows_:
        x_ = Xs[kprev]; out = x_[:192]; h = x_[192:960].reshape(4, 192)
        hf = h * np.exp(-dt * flow_k)
        ctx = out + ln(hf.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.array([np.log(dt) * 0.1]), 'gap')
        z = mlp(ctx, 'z_prior')[:32]
        q = ln(ctx + lin(z, 'z_head') + W['type_emb.weight'][typ], W['query_norm.weight'], W['query_norm.bias'])
        c = CODE[ev]; has = 0.0 if np.isnan(lv) else 1.0
        zl = 0.0 if not has else zsc(ev, 'value', lv); lb = 0 if not has else fbin(ev, 'value', lv)
        ag = 0.0 if not has else np.log1p(la / 3600.0) / 10
        f8 = feats_fn(has, ag, zl)
        wc = q + W['embed.code.weight'][c] + lin(np.array(f8, float), 'last_numeric')
        if not drop_vbin: wc = wc + W['embed.bins.value.weight'][lb]
        if not drop_rate0: wc = wc + W['embed.bins.rate.weight'][0]
        lg = mlp(wc, 'attr_heads.value_bin') + W['embed_transition.value_bin.weight'][c * 17 + lb]
        p = np.exp(lg - lg.max()); p /= p.sum(); out_p.append(np.sum(p[1:] * CEN[ev]))
    return np.array(out_p)
print('rate0 row norm', np.linalg.norm(W['embed.bins.rate.weight'][0]), 'vbin0', np.linalg.norm(W['embed.bins.value.weight'][0]))
print('last_numeric col norms', np.round(np.linalg.norm(W['last_numeric.weight'], axis=0), 2))
FF = {
 'B': lambda h_, a_, z_: [h_, a_, z_, h_, 0, 0, 0, 0],
 'h-a-z-h-0-0-0-1?': lambda h_, a_, z_: [h_, a_, z_, h_, 0, 0, 0, 1 - h_],
 'h,a,z,0,h,0,0,0': lambda h_, a_, z_: [h_, a_, z_, 0, h_, 0, 0, 0],
 'h,a,z,0,0,0,0,0': lambda h_, a_, z_: [h_, a_, z_, 0, 0, 0, 0, 0],
 'h,a,z,h,0,1,0,1': lambda h_, a_, z_: [h_, a_, z_, h_, 0, 1, 0, 1],
 'h,a,z,h,0,0,0,0 age*': lambda h_, a_, z_: [h_, a_ * 10, z_, h_, 0, 0, 0, 0],
}
for nm_, fn_ in FF.items():
    report(predict3(feats_fn=fn_), nm_)
report(predict3(feats_fn=FF['B'], drop_rate0=True), 'B drop rate0')
report(predict3(feats_fn=FF['B'], drop_vbin=True), 'B drop vbin')
report(predict3(feats_fn=FF['B'], drop_rate0=True, drop_vbin=True), 'B drop both')
```