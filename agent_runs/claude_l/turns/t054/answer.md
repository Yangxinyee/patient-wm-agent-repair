Two findings shrink the search. The "absent" index-0 rows are exactly zero in the ekind, text, bin and categorical tables. Many index-1 rows sit at an untrained scale of about 0.27, so those categories never occur. `embed.kq` row 0 is clearly trained (norm 0.65) and its other rows are not, so every atom probably carries `kq[0]`.

So the hospital-admission atom's vector is `code[11] + kind[0] + ekind[3] + kq[0] + numeric.bias`. The flow timescales run from 10 minutes to 2 years as documented.

Next I brute-force the remaining unknowns for the deterministic first bundle:
- the activation of each MLP (encoder mlp1 and mlp2, `z_post`, `gap`);
- the gap input at the start of a stay;
- whether the posterior is the raw `z_post` output or the prior plus an offset.

The initial h and out are taken as zero. I compare against the true z from the states. An exact match (error around 1e-5) fixes these conventions. A floor well above zero means some structural assumption is still wrong, and the best combinations will show which.

```python
import itertools, math
T64 = lambda n_: load(n_).double()
ACT = {'gelu': F.gelu, 'relu': F.relu, 'silu': F.silu, 'tanh': torch.tanh, 'gelu_t': lambda x: F.gelu(x, approximate='tanh')}
def mlp2l(x, pre, act):  # Sequential(Linear, act, Linear)
    return F.linear(act(F.linear(x, T64(pre+'.0.weight'), T64(pre+'.0.bias'))), T64(pre+'.2.weight'), T64(pre+'.2.bias'))
def atomvec(a, kq=True):
    v = T64('embed.code.weight')[a['code']] + T64('embed.kind.weight')[a['kind']] + T64('embed.ekind.weight')[a['ekind']]
    if kq: v = v + T64('embed.kq.weight')[0]
    v = v + T64('embed.categorical.unit.weight')[a['unit']] + T64('embed.categorical.rate_unit.weight')[a['rate_unit']]
    v = v + T64('embed.bins.value.weight')[a['vb']] + T64('embed.bins.dose.weight')[a['db']] + T64('embed.bins.rate.weight')[a['rb']]
    v = v + T64('embed.numeric.weight') @ torch.tensor(a['num'], dtype=torch.float64) + T64('embed.numeric.bias')
    return v
def setenc(atoms, pre, a1, a2, kq=True):
    n_ = len(atoms); X_ = torch.stack([atomvec(a, kq) for a in atoms])
    s_ = sum(mlp2l(X_[i], pre+'.mlp1', a1) for i in range(n_)) / math.sqrt(n_) + F.linear(torch.tensor([math.log(1+n_)], dtype=torch.float64), T64(pre+'.size.weight'), T64(pre+'.size.bias'))
    return F.layer_norm(mlp2l(s_, pre+'.mlp2', a2), (192,), T64(pre+'.norm.weight'), T64(pre+'.norm.bias'), 1e-5)
z0_true = zS[first[0]].double()
atoms0 = BUN[batch1[0]][0][1]
onb = T64('core.out_norm.bias')
best_ = []
gvals = torch.linspace(-1.0, 2.0, 121, dtype=torch.float64)
for a1n, a2n in itertools.product(ACT, ACT):
    for kq in [True, False]:
        e0 = setenc(atoms0, 'encode', ACT[a1n], ACT[a2n], kq)
        for gan in ACT:
            G_ = mlp2l(gvals[:, None], 'gap', ACT[gan])            # (121,192)
            ctx_ = onb[None] + G_
            for zn in ACT:
                inp_ = torch.cat([ctx_, e0[None].expand(121, -1), torch.zeros(121, 192, dtype=torch.float64)], 1)
                zm = mlp2l(inp_, 'z_post', ACT[zn])[:, :32]
                err1 = (zm - z0_true).abs().max(1).values
                zp_ = mlp2l(ctx_, 'z_prior', ACT[zn])[:, :32]
                err2 = (zm + zp_ - z0_true).abs().max(1).values
                for var, er in [('post', err1), ('prior+post', err2)]:
                    i_ = int(er.argmin()); best_.append((er[i_].item(), a1n, a2n, kq, gan, zn, var, gvals[i_].item()))
best_.sort(key=lambda r: r[0]); print(len(best_)); 
for r_ in best_[:12]: print(r_)
```