`step_local` returns (ctx, z, h_new, out_new) in about 2 ms, so a 192-direction scan at dozens of bundles takes under a minute. There are 1,301 insulin bundles in the fwd_3 stays whose next bundle has a finger-glucose reading.

Next I'll compute the next-reading glucose and MAP predictions from a stepped state: flow decay, then `ctx_from`, then `value_mean`, with the last-value register from `build_reg`. I'll test how the glucose prediction responds to random z-token additions of norm 1, 3 and 10. A response that grows roughly linearly to several mg/dL means the z token is a usable entry point. A response that stays under about 1 mg/dL even at norm 10 means the pre-norm prelude blocks it, and I'd need to edit the prelude attention as well.

```python
LV, LT = build_reg(INFO)
fa = WT['core.flow.a']
def next_pred(k, ztok_add=None, codes=('glucose_finger','map_arterial')):
    c, z, h, o = step_local(k, ztok_add=ztok_add) if ztok_add is not None else step_local(k)
    dt1 = dts[k+1]; hf = h*torch.exp(-dt1*torch.exp(fa)); cv = ctx_from(o, hf, dt1)
    out = []
    for cd in codes:
        lv = LV[cd][k+1]; la = (INFO[k+1]['t']-LT[cd][k+1])*60.0
        out.append(value_mean(cv, [cd], lv, la))
    return out
k = ks_ins[5]; b = next_pred(k); print('base', b)
g = torch.Generator().manual_seed(0); dv = torch.randn(192, generator=g, dtype=fa.dtype); dv/=dv.norm()
for s in [1,3,10,30]: print(s, next_pred(k, ztok_add=s*dv))
```