The first-bundle z is close but not exact. My best candidate correlates 0.96 with the true z (slope 0.75), while setting e to zero drops the correlation to 0.36. So the overall structure (atom embedding → set encoder → z_post on [ctx, e, u]) looks right, and some term in ctx or in the atom vector is off.

The most uncertain term is the first bundle's gap input, log(dt)/10, which has no natural value without a previous bundle. I'll scan it continuously over a wide range for each activation, with and without kq. A sharp minimum near zero error at one value would confirm the structure and pin the convention. A floor well above zero would point to a remaining mistake in the atom vector or encoder.

```python
res = []
for actn in ['silu','gelu']:
    act = ACTS[actn]
    for kq in [True, False]:
        e1 = set_encode(atom_vec(11,0,3,use_kq=kq)[None], 'encode', act)
        for gin in np.linspace(-4, 4, 161):
            g = mlp(torch.tensor([gin], dtype=torch.float32), 'gap', act)
            for ctxn, base_ in [('onb', Wt['core.out_norm.bias']), ('0', torch.zeros(192))]:
                zz = mlp(torch.cat([base_+g, e1, torch.zeros(192)]), 'z_post', act)[:32]
                res.append(((zz-z_true).abs().max().item(), actn, kq, round(gin,2), ctxn))
res.sort(key=lambda t:t[0]); print(res[:10])
```