With the corrected units, the best combination is now GELU throughout. That is the natural design, so I take it as the true activation. The other settings are: `kq` used, √n pooling, and u for a bundle without actions set to the encoded empty set.

The mean error is 0.0106, but one element is still off by 0.42, so something remains wrong in some bundles. As before, I'll list the per-stay errors and inspect the best and worst bundles:
- If the lone ICU-admission bundle is now exact while value bundles are not, the remaining issue is in value atoms.
- If even that bundle is off, the issue is in the empty-u encoding or in ctx.

```python
E_g = torch.stack([setenc_v(aa, 'encode', ACT['gelu'], ACT['gelu']) for aa in atoms1u])
U_g = empty_enc('encode_action', ACT['gelu'])[None].expand_as(E_g)
ctx_g = variants['full_flow'] + mlp2l(torch.tensor(np.log(dts)/10)[:, None], 'gap', ACT['gelu'])
zz_ = mlp2l(torch.cat([ctx_g, E_g, U_g], 1), 'z_post', ACT['gelu'])[:, :32]
er_s = (zz_ - z1_true).abs(); per = er_s.mean(1).numpy()
print('per-stay err quantiles', np.round(np.quantile(per, [0, .1, .25, .5, .75, .9, 1]), 5))
for w_ in list(np.argsort(-per)[:6]) + list(np.argsort(per)[:4]):
    print(batch1[w_], 'err %.4f' % per[w_], 'maxdim', int(er_s[w_].argmax()), '%.3f' % er_s[w_].max().item(), 'dt', dts[w_], [(inv_code[a_['code']], a_['vb'], round(a_['num'][0], 3)) for a_ in atoms1u[w_]])
```