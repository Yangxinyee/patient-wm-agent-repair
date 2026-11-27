The original model's finger-glucose predictions under-use the previous reading. On sel1, its residuals rise with the previous finger value within tier: about +5.3 mg/dL per 50 mg/dL (se 1.4), strongest in tiers 0 and 2.

A well-fitted model would leave no such slope, so its persistence looks too weak. That matters twice:
- Correcting it would lower factual NLL.
- In the rollout, a dose-driven shift would carry forward further, so a smaller insulin gain, with less readout noise, could reach the true ×2 effect.

There is a cheap way to apply it. The value head adds a per-code, per-last-bin logit offset (`embed_transition.value_bin`). So a correction that depends on the last finger value can go into that table without repurposing any units.

First I'll check that the effect is real and stable. On the 500 train stays I'll compute the original model's mean finger residual by last-value bin and by tier, then compare with sel1. A consistent rising pattern would justify the edit; an inconsistent one would kill the idea.

```python
rows_tr = []
for i, (Xs_i, Rr_i) in enumerate(datasets):
    d_ = Rr_i[['code','obs','lvb','zl','has','age']].copy()
    d_['pm'] = np.asarray(base_out[i][0], float); d_['res'] = d_.obs.astype(float) - d_.pm
    rows_tr.append(d_)
RT = pd.concat(rows_tr, ignore_index=True)
cen6 = np.asarray(VOCAB['value_bin_centres'][list(VOCAB['value_bin_centres'].keys())[0]] if isinstance(VOCAB['value_bin_centres'], dict) else VOCAB['value_bin_centres'])
print('centres type', type(VOCAB['value_bin_centres']), (list(VOCAB['value_bin_centres'].keys())[:8] if isinstance(VOCAB['value_bin_centres'], dict) else np.shape(VOCAB['value_bin_centres'])))
for nm_, D_ in [('train', RT), ('sel1', R1)]:
    s_ = D_[(D_.code==6)]
    g_ = s_.groupby('lvb').res.agg(['size','mean']).round(2)
    print(nm_, 'gf rows', len(s_), 'mean res %.2f' % s_.res.mean())
    print('  ', ' '.join(f"b{int(b)}:{m:+.1f}({int(n)})" for b,(n,m) in g_.iterrows()))
```