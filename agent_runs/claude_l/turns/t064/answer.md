The empty-set encoding for u helps again: the error is now 0.040. GELU fits best in nearly every slot, including the `z_post` activation that ReLU had won before. It is still not exact, though, and a single error of about 1.37 recurs in every variant.

A recurring outlier like that more likely means a few bundles whose atoms I built wrongly than a structural defect. If most stays already match almost exactly, the remaining problem is in the data, not the architecture.

Next I'll look at the per-stay error distribution under the best combination, then inspect the worst cases: their atoms, recorded values, bundle times and dt.

```python
E_g = torch.stack([setenc_v(aa, 'encode', ACT['gelu'], ACT['relu']) for aa in atoms1])
U_g = empty_enc('encode_action', ACT['gelu'])[None].expand_as(E_g)
ctx_g = variants['full_flow'] + mlp2l(torch.tensor(np.log(dts)/10)[:, None], 'gap', ACT['gelu'])
zz_ = mlp2l(torch.cat([ctx_g, E_g, U_g], 1), 'z_post', ACT['gelu'])[:, :32]
er_s = (zz_ - z1_true).abs()
per = er_s.mean(1).numpy(); print('per-stay err quantiles', np.round(np.quantile(per, [0, .1, .25, .5, .75, .9, 1]), 4))
worst = np.argsort(-per)[:5]
for w_ in worst:
    print(batch1[w_], 'err %.3f' % per[w_], 'max dim', int(er_s[w_].argmax()), 'hr', hrv[sec_idx[w_]], 'dt', dts[w_], [(a_['code'], a_['vb'], round(a_['num'][0], 3)) for a_ in atoms1[w_]])
best_ = np.argsort(per)[:3]
for w_ in best_: print('best', batch1[w_], 'err %.4f' % per[w_], [(a_['code'], a_['vb'], round(a_['num'][0], 3)) for a_ in atoms1[w_]])
```