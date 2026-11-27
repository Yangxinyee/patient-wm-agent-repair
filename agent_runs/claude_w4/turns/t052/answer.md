GELU now fits better, but the copy is still inexact: maximum z_1 error 0.63, mean 0.058. Bundle 1 holds several different reading combinations, so I can find where the mismatch sits by splitting the error:
- If single-atom bundles match and multi-atom bundles do not, the set aggregation is wrong.
- If one code family fails, the per-code encoding is wrong.

I will also test a few alternative aggregation and size conventions (sum/√n vs mean vs sum; log(1+n) vs n) and the tanh-approximate GELU. If no variant gets near exact, I will stop building the full copy and use the forward tool to measure the model's responses directly.

```python
def set_enc2(atoms, p, act, agg='sqrt', size='log1p'):
    n = atoms.shape[0]; m1 = mlp(atoms, p+'.mlp1', act)
    s_ = {'sqrt': m1.sum(0)/np.sqrt(n), 'mean': m1.mean(0), 'sum': m1.sum(0)}[agg]
    sz = {'log1p': np.log1p(n), 'log': np.log(n), 'n': float(n)}[size]
    return ln(mlp(s_ + lin(torch.tensor([sz],dtype=torch.float32), p+'.size'), p+'.mlp2', act), p+'.norm')
comp = [tuple(sorted(e for e,_ in b)) for b in b1]
out_ = []
for an in ['gelu','gelu_tanh']:
  act = acts[an]
  ctx = o0 + ln(h0f.mean(0),'core.out_norm') + mlp(torch.tensor([np.log(3600.0)/10],dtype=torch.float32),'gap',act)
  for agg in ['sqrt','mean','sum']:
    for size in ['log1p','log','n']:
      for kq in [None, 0]:
        unit_idx = {'mmHg': 5, 'mg/dL': 1}
        E1 = torch.stack([set_enc2(torch.stack([meas_atom(ev, v, dict(kq=kq), unit_idx) for ev, v in b]), 'encode', act, agg, size) for b in b1])
        zo = lin(act(lin(torch.cat([ctx[None].expand(100,-1), E1, torch.zeros(100,192)],1),'z_post.0')),'z_post.2')[:, :32]
        err = (zo-Z1).abs().max(1).values.numpy()
        out_.append((err.max(), err.mean(), an, agg, size, kq, err))
out_.sort(key=lambda t: t[1])
for o_ in out_[:4]: print(o_[:6])
err = out_[0][6]
print('Z1 sd across stays (mean over dims):', Z1.std(0).mean().item())
print(pd.DataFrame({'comp': comp, 'err': err}).groupby('comp').err.agg(['mean','max','count']).round(4).to_string())
```