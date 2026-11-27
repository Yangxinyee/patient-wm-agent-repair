The identity holds: out equals out_norm(mean over slots of h) to 9e-7. So the states layout and the layer norm (eps ≈ 1e-5) are confirmed.

The previous run had failed before defining the variant-testing function, so I'll redefine it and run the grid: activation function × gap-input scaling, with the context and z posterior as documented. One variant dropping the z error to near zero would pin down both. If none does, the fault lies elsewhere: the concatenation order, the embedding terms or the action/END handling.

```python
acts = {'gelu':F.gelu, 'gelu_tanh':lambda x: F.gelu(x, approximate='tanh'), 'silu':F.silu, 'relu':F.relu}
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
for an, af in acts.items():
    for gm in ['s10','s','h10','m10','h']:
        print(an, gm, z_err2(30, af, gm))
```