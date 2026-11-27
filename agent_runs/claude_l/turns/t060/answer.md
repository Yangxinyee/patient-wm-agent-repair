The block order `[ctx, e, u]` is confirmed as best (0.056); every other order is at least twice as bad. The data also show that bundles are either measurement-only (2,751) or action-only (2,181), never mixed.

`z` is only mildly sensitive to the details I have varied (ctx parts, `kq`, pooling). So the leftover error probably comes from a structural feature of the set encoder I have not yet considered. Candidates are residual connections, in the per-atom MLP (atom + mlp1(atom)) or in the set MLP (s + mlp2(s)). Such a mistake would distort e for every bundle in much the same way.

This run tries those encoder structures with several activations. I also print the `z_post.0` weight norm for each input block, to see how strongly z depends on ctx versus e. A near-exact match for some structure confirms a residual design. If nothing improves, I will move to testing the action encoder on drug bundles and the full state update.

```python
Wz = T64('z_post.0.weight'); print('z_post.0 block norms c/e/u:', [round(Wz[:, 192*i:192*(i+1)].norm().item(), 2) for i in range(3)])
def setenc_v(atoms, pre, a1, a2, res1=False, res2=False, pool='sqrt', kq=True):
    n_ = len(atoms); X_ = torch.stack([atomvec(a, kq) for a in atoms])
    H_ = mlp2l(X_, pre+'.mlp1', a1); H_ = H_ + X_ if res1 else H_
    s_ = H_.sum(0) / (math.sqrt(n_) if pool == 'sqrt' else 1.0) + F.linear(torch.tensor([math.log(1+n_)], dtype=torch.float64), T64(pre+'.size.weight'), T64(pre+'.size.bias'))
    o_ = mlp2l(s_, pre+'.mlp2', a2); o_ = o_ + s_ if res2 else o_
    return F.layer_norm(o_, (192,), T64(pre+'.norm.weight'), T64(pre+'.norm.bias'), 1e-5)
rv = []
for res1, res2 in itertools.product([False, True], [False, True]):
    for a1n in ['gelu', 'relu', 'silu']:
        for a2n in ['gelu', 'relu', 'silu']:
            for pool in ['sqrt', 'sum']:
                E_ = torch.stack([setenc_v(aa, 'encode', ACT[a1n], ACT[a2n], res1, res2, pool) for aa in atoms1])
                for zn in ['relu', 'gelu']:
                    zz_ = mlp2l(torch.cat([ctx_b, E_, torch.zeros_like(E_)], 1), 'z_post', ACT[zn])[:, :32]
                    er = (zz_ - z1_true).abs(); rv.append((er.mean().item(), er.max().item(), res1, res2, a1n, a2n, pool, zn))
rv.sort(key=lambda r: r[0])
for r_ in rv[:8]: print(np.round(r_[:2], 5), r_[2:])
```