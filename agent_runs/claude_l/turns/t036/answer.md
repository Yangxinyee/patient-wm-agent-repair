The output path is confirmed: the reported out equals LayerNorm(mean of the reported h slots) to 1e-6. So the stored h is the post-coda state, and core.out_norm works as described.

The first-bundle z mismatch must therefore come from the inputs to z_post: the initial context, or the bundle encoding.

Next I'll vary the initial state, since "h starts at zero" is my own assumption. I'll try h0 = 0 or core.slots, and out0 = 0 or out_norm(mean h0), together with a scan of the gap input. If one combination drives the error to about 0, the initial state was the problem. If none does, the encoder or atom vector is wrong.

```python
res = []
act = F.silu
for kq in [True, False]:
    e1 = set_encode(atom_vec(11,0,3,use_kq=kq)[None], 'encode', act)
    for h0n, h0 in [('zero', torch.zeros(4,192)), ('slots', Wt['core.slots'])]:
        for out0n in ['zero', 'norm']:
            out0 = torch.zeros(192) if out0n=='zero' else ln(h0.mean(0), 'core.out_norm')
            for gin in np.linspace(-3, 3, 121):
                g = mlp(torch.tensor([gin], dtype=torch.float32), 'gap', act)
                ctx = out0 + ln(h0.mean(0), 'core.out_norm') + g
                zz = mlp(torch.cat([ctx, e1, torch.zeros(192)]), 'z_post', act)[:32]
                res.append(((zz-z_true).abs().max().item(), kq, h0n, out0n, round(gin,3)))
res.sort(key=lambda t:t[0]); print(res[:8])
```