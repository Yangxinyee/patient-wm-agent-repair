The z check failed under every unit mapping: median error about 0.28 for the best one, versus the ~1e-4 an exact match would give. So something beyond the unit table is wrong.

To find it, I'll first test the simplest documented identity, out = out_norm(mean over slots of the new h), directly on the states file. If it holds to rounding error, the states layout and the layer norm are confirmed and the fault is in the context, gap or embedding. I'll also try the activation variants (exact GELU, tanh-GELU, SiLU, ReLU) and dt in seconds versus hours inside the gap term, to see which brings the z error down.

```python
o_chk = (Xs[:, :192] - ln(Xs[:, 192:960].reshape(-1,4,192).mean(1), 'core.out_norm')).abs().max().item()
print('out == out_norm(mean_slots(h)) max err', o_chk)
def z_err2(ntest=40, act=F.gelu, gapmode='s10', decay_on=True, order='cef'):
    global ACT; ACT = act; errs=[]
    for r_, rp, dt, ms_ in tests[:ntest]:
        e_ = encode([meas_atom(c, v) for c, v in ms_])
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a'])) if decay_on else torch.ones(192)
        g_in = {'s10': math.log(dt)/10, 's': math.log(dt), 'h10': math.log(dt/3600)/10, 'm10': math.log(dt/60)/10, 'h': math.log(dt/3600)}[gapmode]
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([g_in]), 'gap')
        parts = {'c': ctx, 'e': e_, 'f': torch.zeros(192)}
        zp = mlp(torch.cat([parts[ch] for ch in order]), 'z_post')[:32]
        errs.append((zp - Xs[r_, 960:992]).abs().max().item())
    return round(float(np.median(errs)),4), round(float(np.max(errs)),4)
UNIT = {'mmHg':2, 'mg/dL':3}
acts = {'gelu':F.gelu, 'gelu_tanh':lambda x: F.gelu(x, approximate='tanh'), 'silu':F.silu, 'relu':F.relu}
for an, af in acts.items():
    for gm in ['s10','s','h10','m10','h']:
        print(an, gm, z_err2(30, af, gm))
```