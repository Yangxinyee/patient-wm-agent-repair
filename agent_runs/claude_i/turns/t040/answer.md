No combination reproduces the first z posterior mean exactly; the best maximum error is 0.62 (ReLU, with kq and numeric bias). So at least one assumption is wrong. Candidates:
- the initial out/h are not zero;
- the visit atom carries extra attributes, such as a has_value flag or its end time;
- a different unit row is used;
- the z_post inputs are ordered or composed differently.

Before guessing further, I'll check how close the best candidate is: its correlation and regression slope with the target. If the pattern is nearly right with a small offset, one additive term is missing. If it is uncorrelated, the structure itself is wrong.

```python
def zpost_cand(act, kq, numb, gs, unit=0, extra=None, ctx_extra=None):
    a = atom_vec(11, 0, 3, kq=kq, numb=numb, unit=unit)
    if extra is not None: a = a + extra
    e = encode_set(a[None], 'encode', act)
    gap = mlp(torch.tensor([[gs]], dtype=torch.float32), 'gap', act)[0]
    ctx = W['core.out_norm.bias'] + gap
    if ctx_extra is not None: ctx = ctx + ctx_extra
    zp = mlp(torch.cat([ctx, e, torch.zeros(192)]), 'z_post', act)
    return zp, e, ctx
for act in ['relu','gelu','silu']:
    zp, e, ctx = zpost_cand(act, True, True, -0.05)
    zm = zp[:32].numpy(); zt = z_target.numpy()
    print(act, "corr", np.corrcoef(zm, zt)[0,1].round(4), "slope", np.polyfit(zm, zt, 1).round(3), " logsd part:", zp[32:40].numpy().round(2))
print("target", z_target.numpy()[:10].round(3))
zp, e, ctx = zpost_cand('relu', True, True, -0.05); print("cand  ", zp[:10].numpy().round(3))
```