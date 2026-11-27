The first-bundle state is identical across all 100 stays (max difference 1.5e-5), so it is a deterministic fixed point I can reproduce exactly. out is LayerNorm-scaled (std ≈ 1). The h slots are large (norms about 400–570), so h is not normalised.

I'll write the building blocks: LayerNorm, MLPs with a choice of activation, and the atom embedding. Then I'll compute the first bundle's z posterior mean, z_post([ctx, e₁, u₁ = 0]), under combinations of:
- activation;
- whether kq[0] and the numeric bias are added;
- the z_post output split;
- the unknown gap input for the first bundle, scanned as a scalar.

The posterior mean depends only on the input side, so an exact match pins down the embedding and encoder.

```python
import torch.nn.functional as Fnn, itertools
torch.set_grad_enabled(False)
def lin(x, n, bias=True):
    y = x @ W[n+'.weight'].T
    if bias and (n+'.bias') in W: y = y + W[n+'.bias']
    return y
def ln(x, n, eps=1e-5):
    return Fnn.layer_norm(x, (x.shape[-1],), W[n+'.weight'], W[n+'.bias'], eps)
ACTS = {'gelu': Fnn.gelu, 'gelu_tanh': lambda x: Fnn.gelu(x, approximate='tanh'), 'relu': Fnn.relu, 'silu': Fnn.silu}
def mlp(x, n, act):  # n.0 -> act -> n.2
    return lin(ACTS[act](lin(x, n+'.0')), n+'.2')
def encode_set(atoms, pre, act):   # atoms: (n,192)
    n_ = atoms.shape[0]
    s = mlp(atoms, pre+'.mlp1', act).sum(0)/np.sqrt(n_) + lin(torch.tensor([[np.log(1+n_)]], dtype=torch.float32), pre+'.size')[0]
    return ln(mlp(s, pre+'.mlp2', act), pre+'.norm')
z_target = torch.tensor(x0[960:])
def atom_vec(code, kind, ekind, kq=True, numb=True, unit=0, runit=0, vb=0, rb=0, db=0, num=None):
    v = W['embed.code.weight'][code] + W['embed.kind.weight'][kind] + W['embed.ekind.weight'][ekind]
    if kq: v = v + W['embed.kq.weight'][0]
    v = v + W['embed.categorical.unit.weight'][unit] + W['embed.categorical.rate_unit.weight'][runit]
    v = v + W['embed.bins.value.weight'][vb] + W['embed.bins.rate.weight'][rb] + W['embed.bins.dose.weight'][db]
    numv = torch.zeros(6) if num is None else torch.tensor(num, dtype=torch.float32)
    v = v + W['embed.numeric.weight'] @ numv + (W['embed.numeric.bias'] if numb else 0)
    return v
best = []
for act, kq, numb in itertools.product(ACTS.keys(), [True, False], [True, False]):
    a = atom_vec(11, 0, 3, kq=kq, numb=numb)
    e = encode_set(a[None], 'encode', act)
    on = W['core.out_norm.bias']  # out_norm of zero vector
    for gs in np.linspace(-1.5, 2.0, 141):
        gap = mlp(torch.tensor([[gs]], dtype=torch.float32), 'gap', act)[0]
        for out0_zero in [True]:
            ctx = on + gap
            zp = mlp(torch.cat([ctx, e, torch.zeros(192)]), 'z_post', act)
            for split in [0, 1]:
                zm = zp[:32] if split==0 else zp[32:]
                err = (zm - z_target).abs().max().item()
                best.append((err, act, kq, numb, gs, split))
best.sort(key=lambda t: t[0]); print(best[:8])
```