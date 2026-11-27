Matching the second bundle gets much closer: mean absolute error 0.056, against z spreads of 0.1 to 0.6. The best combination is GELU in encoder mlp1, ReLU in encoder mlp2, GELU in `gap`, a gap input of log(dt in seconds)/10, and ReLU in `z_post`.

It is still not exact, and the worst bundle is off by 1.36, so one structural detail remains wrong. To find it I will break the best combination's error down by bundle size and by which codes are present:
- If the error grows with the number of atoms, the set-pooling rule (sum/√n, mean, or the size term) is wrong.
- If it is tied to particular codes, that code's atom features are wrong (unit, bins or z-score).
- If it is uniform, the problem is shared, such as ctx or the `kq` term.

```python
def run_combo(a1n, a2n, gan, zn, kq=True, pool='sqrt', gv_=None, atoms_list=atoms1, ctxb=ctx_base):
    def setenc2(atoms, pre):
        n_ = len(atoms); X_ = torch.stack([atomvec(a, kq) for a in atoms])
        H_ = mlp2l(X_, pre+'.mlp1', ACT[a1n]).sum(0)
        H_ = H_ / {'sqrt': math.sqrt(n_), 'mean': n_, 'sum': 1.0}[pool]
        s_ = H_ + F.linear(torch.tensor([math.log(1+n_)], dtype=torch.float64), T64(pre+'.size.weight'), T64(pre+'.size.bias'))
        return F.layer_norm(mlp2l(s_, pre+'.mlp2', ACT[a2n]), (192,), T64(pre+'.norm.weight'), T64(pre+'.norm.bias'), 1e-5)
    E1 = torch.stack([setenc2(aa, 'encode') for aa in atoms_list])
    gv_ = np.log(dts)/10 if gv_ is None else gv_
    ctx_ = ctxb + mlp2l(torch.tensor(gv_)[:, None], 'gap', ACT[gan])
    zm = mlp2l(torch.cat([ctx_, E1, torch.zeros_like(E1)], 1), 'z_post', ACT[zn])[:, :32]
    return zm, E1
for pool in ['sqrt', 'mean', 'sum']:
    for kq in [True, False]:
        zm, _ = run_combo('gelu', 'relu', 'gelu', 'relu', kq, pool)
        er = (zm - z1_true).abs(); print(pool, kq, 'mean %.5f max %.4f' % (er.mean().item(), er.max().item()))
zm, _ = run_combo('gelu', 'relu', 'gelu', 'relu', True, 'sqrt')
er_b = (zm - z1_true).abs().mean(1).numpy()
comp = [tuple(sorted(a_['code'] for a_ in aa)) for aa in atoms1]
dfe = pd.DataFrame({'err': er_b, 'n': [len(a) for a in atoms1], 'comp': comp, 'dt': dts})
print(dfe.groupby('n')['err'].agg(['mean', 'max', 'count']))
print(dfe.groupby('comp')['err'].agg(['mean', 'max', 'count']).sort_values('count', ascending=False).head(10))
print(dfe.groupby('dt')['err'].agg(['mean', 'count']))
```