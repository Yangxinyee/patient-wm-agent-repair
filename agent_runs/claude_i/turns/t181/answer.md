The z error is spread over every bundle type, and it is largest (about 2.4) on each stay's first bundle. That points to the stored `ctx` tensor being stale or wrong rather than to the atom recipe. I'll recompute ctx directly from the tool's states with the verified formula: previous out, plus out_norm of the time-decayed mean slot state, plus the gap term. The first bundle starts from a zero state with dt clamped to 1 s.

If the z error drops to about 0, the new encoder is exact. If it stays, the recipe itself is wrong.

```python
X3t = torch.tensor(X3, dtype=torch.float64)
fa_ = T64('core.flow.a')
def gap2(dt_s):
    g = torch.log(torch.clamp(torch.as_tensor(dt_s, dtype=torch.float64), min=1.0)).reshape(-1,1)/10.0
    return gelu_(g @ T64('gap.0.weight').T + T64('gap.0.bias')) @ T64('gap.2.weight').T + T64('gap.2.bias')
first_ = np.r_[True, S3[1:]!=S3[:-1]]
outp = torch.zeros((5156,192), dtype=torch.float64); hp = torch.zeros((5156,4,192), dtype=torch.float64)
outp[1:] = X3t[:-1,:192]; hp[1:] = X3t[:-1,192:960].reshape(-1,4,192)
ft = torch.tensor(first_)
outp[ft]=0; hp[ft]=0
dtt = dts.double().clone()
hf_ = hp*torch.exp(-dtt.reshape(-1,1,1)*torch.exp(fa_))
ctx2 = outp + F_.layer_norm(hf_.mean(1),(192,),T64('core.out_norm.weight'),T64('core.out_norm.bias'),1e-5) + gap2(dtt)
zz3,_ = zpost2(ctx2, torch.tensor(Eall), torch.tensor(Uall))
d3 = np.abs(zz3[:,:32].numpy()-X3[:,960:992]).max(1)
print('ctx diff', (ctx2-ctx.double()).abs().max().item(), 'z err median', np.median(d3).round(5), '90%', np.quantile(d3,0.9).round(4), 'max', d3.max().round(3))
print('dts first', dts[:4], 'first dt', dtt[ft][:3])
```