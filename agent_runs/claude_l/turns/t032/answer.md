No variant reproduces the first-bundle z (best max error 1.6, with SiLU), so at least one convention is still wrong.

Before guessing further, I'll check the scale of the true z against my candidates. A similar scale with a different pattern would point to a wrong input (atom features, ctx, or token order). A very different scale would point to a structural error in z_post, for example how mean and log-sd are split.

I'll also test a few more first-bundle variants:
- ctx without the out_norm(mean h) term;
- an e_1 that skips the final norm;
- swapping the order of the [ctx, e, u] blocks.

Whichever variant brings the error near zero tells me where the mistake is.

```python
print("z_true", z_true.numpy().round(3))
act = F.silu
e1 = set_encode(atom_vec(11,0,3,use_kq=False)[None], 'encode', act)
g0 = mlp(torch.tensor([0.0]), 'gap', act)
cands = {}
for actn, act in ACTS.items():
    for kq in [True, False]:
        e1 = set_encode(atom_vec(11,0,3,use_kq=kq)[None], 'encode', act)
        for gin in [0.0, -2.0, 2.0]:
            g = mlp(torch.tensor([gin]), 'gap', act)
            for ctx_name, ctx in [('onb+g', Wt['core.out_norm.bias']+g), ('g', g), ('onb', Wt['core.out_norm.bias']), ('0', torch.zeros(192))]:
                for order in ['ceu','cue','ecu']:
                    parts = {'c': ctx, 'e': e1, 'u': torch.zeros(192)}
                    inp = torch.cat([parts[o] for o in order])
                    zz = mlp(inp, 'z_post', act)
                    for half in [0, 32]:
                        cands[(actn, kq, gin, ctx_name, order, half)] = (zz[half:half+32]-z_true).abs().max().item()
srt = sorted(cands.items(), key=lambda kv: kv[1]); print(srt[:10])
```