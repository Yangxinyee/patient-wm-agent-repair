Using kq row 0 together with the corrected action vector halves the error again. The best combination, mmHg = unit row 2 and mg/dL = row 3, gives a median max error of 0.017. That is still not exact.

The remaining fault may be code-specific: arterial and cuff MAP, or finger and lab glucose, could use different unit rows. So I'll measure the error separately on bundles holding a single code. One code staying wrong while the others match would point to that code's embedding.

```python
UNIT = {'mmHg':2, 'mg/dL':3}
def z_err_single(code_filter, kq=0, unit_override=None, ntest=200):
    errs=[]
    for r_, rp, dt, ms_ in tests[:ntest]:
        cs = set(c for c,_ in ms_)
        if cs != {code_filter}: continue
        atoms=[]
        for c, v in ms_:
            x = Wt['embed.code.weight'][code_id[c]] + Wt['embed.kind.weight'][0] + Wt['embed.ekind.weight'][1] + Wt['embed.kq.weight'][kq]
            ui = unit_override if unit_override is not None else UNIT['mmHg' if c.startswith('map') else 'mg/dL']
            x = x + Wt['embed.categorical.unit.weight'][ui] + Wt['embed.bins.value.weight'][fbin(c,'value',v)]
            x = x + Wt['embed.numeric.weight'] @ torch.tensor([zsc(c,'value',v), 1., 0., 0., 0., 0.]) + Wt['embed.numeric.bias']
            atoms.append(x)
        e_ = encode(atoms)
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
        zp = mlp(torch.cat([ctx, e_, u_empty]), 'z_post')[:32]
        errs.append((zp - Xs[r_, 960:992]).abs().max().item())
    return (round(float(np.median(errs)),5), len(errs)) if errs else None
for c_ in ['map_arterial','map_cuff','glucose_finger']:
    print(c_, [ (ui, z_err_single(c_, unit_override=ui)) for ui in [0,1,2,3,4,5]])
```