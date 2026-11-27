This is real progress. My z_post's first 32 outputs track the model's z dimension by dimension (overall correlation 0.87, most dimensions 0.6–0.98). So the output mapping and network are right, but one input is still slightly wrong.

To locate it, I'll split the error by bundle features: atom count n, which codes the bundle holds, and the gap dt. If the error grows with n, the set aggregation (sum/√n, the size term) is off. If it depends on code, the atom embedding is off for those codes. If it tracks dt, the time-flow or gap formula is off. I'll also drop the kq term, which the architecture doc doesn't list and which slightly improved the fit.

```python
def meas_atom2(code, val, kq=None, unit_tab=True, cat_extra=None):
    x = Wt['embed.code.weight'][code_id[code]] + Wt['embed.kind.weight'][0] + Wt['embed.ekind.weight'][1]
    if kq is not None: x = x + Wt['embed.kq.weight'][kq]
    if unit_tab: x = x + Wt['embed.categorical.unit.weight'][UNIT['mmHg' if code.startswith('map') else 'mg/dL']]
    if cat_extra:
        for tb, ix in cat_extra: x = x + Wt['embed.categorical.%s.weight'%tb][ix]
    x = x + Wt['embed.bins.value.weight'][fbin(code,'value',val)]
    return x + Wt['embed.numeric.weight'] @ torch.tensor([zsc(code,'value',val), 1., 0., 0., 0., 0.]) + Wt['embed.numeric.bias']
def per_test_err(ntest=200, **kw):
    out_=[]
    for r_, rp, dt, ms_ in tests[:ntest]:
        e_ = encode([meas_atom2(c, v, **kw) for c, v in ms_])
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
        zp = mlp(torch.cat([ctx, e_, torch.zeros(192)]), 'z_post')[:32]
        out_.append(((zp - Xs[r_, 960:992]).abs().mean().item(), len(ms_), dt, '+'.join(sorted(c for c,_ in ms_))))
    return pd.DataFrame(out_, columns=['err','n','dt','codes'])
D0 = per_test_err(200)
print(D0.groupby('n').err.agg(['mean','count']).round(4).to_string())
print(D0.groupby('codes').err.agg(['mean','count']).round(4).to_string())
print(D0.groupby(pd.cut(D0.dt,[0,600,3000,3400,3700,1e6])).err.agg(['mean','count']).round(4).to_string())
```