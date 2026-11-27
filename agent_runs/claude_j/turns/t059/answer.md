Scanning the gap input doesn't fix it. The best-fitting value wanders between -0.25 and 0.85 and the minimum error stays at 0.025–0.06. So the gap term is not the main fault; something structural in the context or inputs is.

Next I'll test structural variants of the z input:
- dropping the previous out from the context;
- doubling the normed-h term;
- using h without decay;
- a constant (encoder-of-empty-set) action vector instead of zeros;
- an off-by-one in which state row counts as "previous".

A variant that drops the error several-fold would show which documented formula I'm misreading.

```python
def z_err_v(variant, ntest=60):
    errs=[]
    u_empty = ln(mlp(lin(torch.tensor([0.0]), 'encode_action.size'), 'encode_action.mlp2'), 'encode_action.norm')
    for r_, rp, dt, ms_ in tests[:ntest]:
        if variant=='shift+1': r_, rp = r_+1, rp+1
        if variant=='shift-1': r_, rp = r_-1, rp-1
        e_ = encode([meas_atom2(c, v) for c, v in ms_])
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        hn = ln((h_p*dec).mean(0), 'core.out_norm'); g = mlp(torch.tensor([math.log(dt)/10]), 'gap')
        u_ = torch.zeros(192)
        if variant=='noout': ctx = hn + g
        elif variant=='2hn': ctx = 2*hn + g
        elif variant=='uempty': ctx = out_p+hn+g; u_ = u_empty
        else: ctx = out_p + hn + g
        zp = mlp(torch.cat([ctx, e_, u_]), 'z_post')[:32]
        errs.append((zp - Xs[r_, 960:992]).abs().mean().item())
    return round(float(np.mean(errs)),4)
for v_ in ['base','noout','2hn','uempty','shift+1','shift-1']:
    print(v_, z_err_v(v_))
```