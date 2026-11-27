Every failing bundle contains glucose_lab, so glucose_lab's embedding differs from glucose_finger's despite the same mg/dL unit. I'll scan glucose_lab's unit row (0–5) together with the unused-looking extras that a lab result might carry: status row 1, text row 1, other kq rows. A combination giving near-zero error on these 89 bundles would complete the measurement embedding.

```python
def meas_atom_gl(code, val, gl_unit=3, gl_extra=None, gl_kq=0):
    if code!='glucose_lab': return meas_atom(code, val)
    x = Wt['embed.code.weight'][code_id[code]] + Wt['embed.kind.weight'][0] + Wt['embed.ekind.weight'][1] + Wt['embed.kq.weight'][gl_kq]
    x = x + Wt['embed.categorical.unit.weight'][gl_unit]
    if gl_extra: 
        for tb, ix in gl_extra: x = x + (Wt['embed.%s.weight'%tb][ix] if tb in ('text',) else Wt['embed.categorical.%s.weight'%tb][ix])
    x = x + Wt['embed.bins.value.weight'][fbin(code,'value',val)]
    return x + Wt['embed.numeric.weight'] @ torch.tensor([zsc(code,'value',val), 1., 0., 0., 0., 0.]) + Wt['embed.numeric.bias']
def err_bad(**kw):
    ee=[]
    for i_, _ in bad[:25]:
        r_, rp, dt, ms_ = tests[i_]
        e_ = encode([meas_atom_gl(c, v, **kw) for c, v in ms_])
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
        zp = mlp(torch.cat([ctx, e_, u_empty]), 'z_post')[:32]
        ee.append((zp - Xs[r_, 960:992]).abs().max().item())
    return round(float(np.median(ee)),6)
for gu in range(6):
    print('gl_unit', gu, err_bad(gl_unit=gu), 'status1', err_bad(gl_unit=gu, gl_extra=[('status',1)]), 'text1', err_bad(gl_unit=gu, gl_extra=[('text',1)]))
for kq_ in [1,2,3]:
    print('kq', kq_, [err_bad(gl_unit=gu, gl_kq=kq_) for gu in range(6)])
```