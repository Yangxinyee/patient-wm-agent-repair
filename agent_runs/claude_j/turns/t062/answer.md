Exact matches, to five decimals, for all three codes. Both arterial and cuff MAP use unit row 2. Glucose_finger matches only with unit index 0, the all-zero row, so mg/dL readings effectively get no unit embedding. Measurements also get kq row 0.

I'll confirm this mapping on all 493 measurement-only bundles, including the mixed ones with glucose_lab. Then I'll start on the action encoder: insulin and norepinephrine start and END atoms in bundles where the action vector is non-empty. I'll print one such bundle's atoms so I can choose the hypotheses to test. A near-zero error on all measurement-only bundles would close out the measurement embedding.

```python
UNIT = {'mmHg':2, 'mg/dL':0}
def meas_atom(code, val):
    x = Wt['embed.code.weight'][code_id[code]] + Wt['embed.kind.weight'][0] + Wt['embed.ekind.weight'][1] + Wt['embed.kq.weight'][0]
    x = x + Wt['embed.categorical.unit.weight'][UNIT['mmHg' if code.startswith('map') else 'mg/dL']]
    x = x + Wt['embed.bins.value.weight'][fbin(code,'value',val)]
    return x + Wt['embed.numeric.weight'] @ torch.tensor([zsc(code,'value',val), 1., 0., 0., 0., 0.]) + Wt['embed.numeric.bias']
errs=[]
for r_, rp, dt, ms_ in tests:
    e_ = encode([meas_atom(c, v) for c, v in ms_])
    out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
    dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
    ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
    zp = mlp(torch.cat([ctx, e_, u_empty]), 'z_post')[:32]
    errs.append((zp - Xs[r_, 960:992]).abs().max().item())
print('all meas-only bundles: median %.2e max %.2e' % (np.median(errs), np.max(errs)), len(errs))
# action bundles to test next
bb0 = build_bundles(tu[tu.stay_id==stays_u[0]]); k0 = sorted(bb0.keys())
print([(e2,k2, getattr(r2,'rate'), getattr(r2,'dose')) for e2,k2,r2 in bb0[k0[13]]])
```