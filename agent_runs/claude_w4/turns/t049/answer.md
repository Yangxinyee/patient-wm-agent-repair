No match yet: the best maximum error on bundle 0's z is 1.21, far from exact. One thing is settled. Adding the categorical rows or the absent-bin rows makes no difference at all, so those tables have zero padding rows and absent attributes add nothing. `embed.kq` does matter.

Remaining suspects: the input order to `z_post`, the gap input for the first bundle (unknown dt handling), the activation, and atom attributes I am not including. The visit row also carries an end_hour, perhaps a `discharged_to`-type field.

Next I will reread /data/README.md for any extra atom attributes. I will also widen the search:
- all 6 orders of [ctx, e, u];
- a fine grid over the gap input scalar;
- both halves of the z_post output;
- the activation;
- the `kq` index.

An error near 1e-4 for some combination would pin the convention. If the minimum stays around 1, the encoding itself (the atom attributes) is wrong.

```python
print(open('/data/README.md').read()[:3000])
grid = np.linspace(-4, 4, 161)
res_ = []
for an, act in acts.items():
  for kq in [None,0,1,2,3]:
    e0 = set_enc(atom_emb(11, 0, 3, dict(kq=kq))[None], 'encode', act)
    G = mlp(torch.tensor(grid, dtype=torch.float32)[:,None], 'gap', act)   # (161,192)
    for octx in ['bias','2bias','zero']:
      base_ctx = {'bias': W['core.out_norm.bias'], '2bias': 2*W['core.out_norm.bias'], 'zero': torch.zeros(192)}[octx]
      for useg in [True, False]:
        ctxs = base_ctx[None] + (G if useg else 0*G)
        for perm in itertools.permutations(range(3)):
          parts = [ctxs, e0[None].expand(len(grid),-1), torch.zeros(len(grid),192)]
          inp = torch.cat([parts[i] for i in perm], 1)
          zo = lin(act(lin(inp, 'z_post.0')), 'z_post.2')
          for half in [0, 1]:
            err = (zo[:, 32*half:32*half+32] - zt).abs().max(1).values
            i = int(err.argmin()); res_.append((float(err[i]), an, kq, octx, useg, perm, half, float(grid[i])))
res_.sort(key=lambda t: t[0])
for b_ in res_[:10]: print(b_)
```