That's the main fault. With the action vector set to the action encoder applied to an empty set (a constant vector) instead of zero, the error drops nearly five-fold, from 0.049 to 0.0106.

To reach an exact match, I'll redo the earlier side-tests with the corrected action vector: the unit-table row mapping, with or without the kq term, and the gap-input scaling. A combination with error near 1e-4 would finish this part of the replica. If 0.01 remains everywhere, there is another small structural detail still to find.

```python
u_empty = ln(mlp(lin(torch.tensor([0.0]), 'encode_action.size'), 'encode_action.mlp2'), 'encode_action.norm')
e_empty = ln(mlp(lin(torch.tensor([0.0]), 'encode.size'), 'encode.mlp2'), 'encode.norm')
def z_err4(ntest=60, kq=None, units=(2,3), gscale='s10', act=F.gelu):
    global ACT, UNIT; ACT = act; UNIT = {'mmHg':units[0], 'mg/dL':units[1]}
    errs=[]
    for r_, rp, dt, ms_ in tests[:ntest]:
        e_ = encode([meas_atom2(c, v, kq=kq) for c, v in ms_])
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        g_in = math.log(dt)/10 if gscale=='s10' else math.log(dt/60)/10
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([g_in]), 'gap')
        zp = mlp(torch.cat([ctx, e_, u_empty]), 'z_post')[:32]
        errs.append((zp - Xs[r_, 960:992]).abs().max().item())
    return round(float(np.median(errs)),5)
for un in itertools.permutations([2,3,4,5],2):
    print(un, z_err4(30, units=un), 'kq0', z_err4(30, kq=0, units=un))
print('m10', z_err4(30, gscale='m10'), 'gelu_tanh', z_err4(30, act=lambda x: F.gelu(x, approximate='tanh')))
ACT = F.gelu; UNIT = {'mmHg':2, 'mg/dL':3}
```