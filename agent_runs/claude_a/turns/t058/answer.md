Close, but not exact. The best convention puts z_value in column 0 and has_value in column 1, as documented. It uses unit row 3 for mg/dL and row 2 for mmHg, and adds the kind, event-kind, kq and value-bin rows. It still leaves an error of 0.013 on single-atom bundles and 0.05 on multi-atom bundles.

Next I will break the single-atom error down by code and by value. I will also test the aggregation on multi-atom bundles: 1/√n against 1/n, and the size term with n against log(1 + n). If the single-atom error sits in one code, that code's unit row or normalisation is wrong. If it grows with |z|, the clipping or the bins are off. One of the aggregation variants should make the multi-atom error drop to the single-atom level.

```python
e1_best = ln(mlp(mlp(atomvec1(**{k: MEAS_CFG[k] for k in ['cz','ch','ug','um','ak','ae','aq','avb']}), 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm')
er1 = (e1_best[:, :32] + u_empty[:, :32] - r1).abs().mean(1).numpy()
D1 = pd.DataFrame({'event': sub1.event.values, 'value': sub1.value.values, 'err': er1, 'z': zv1, 'vb': vb1})
print(D1.groupby('event').err.agg(['mean', 'max', 'count']).round(4).to_string())
print(D1.groupby('vb').err.agg(['mean', 'count']).round(4).T.to_string())
print(D1.sort_values('err').tail(5).to_string())
# alternative unit assignments per code
for ul in [2, 3, 4, 5]:
    for ucode in ['glucose_lab', 'map_cuff']:
        m_ = (sub1.event == ucode).values
        if m_.sum() == 0: continue
        umap_ = {'glucose_finger': 3, 'glucose_lab': 3, 'map_arterial': 2, 'map_cuff': 2}; umap_[ucode] = ul
        ut_ = P['embed.categorical.unit.weight'][torch.tensor(sub1.event.map(umap_).values)]
        nf = torch.zeros(len(sub1), 6); nf[:, 0] = zv1t; nf[:, 1] = 1
        av_ = P['embed.code.weight'][c1] + P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + P['embed.kq.weight'][0] + ut_ + P['embed.bins.value.weight'][torch.tensor(vb1)] + lin(nf, 'embed.numeric')
        e_ = ln(mlp(mlp(av_, 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm')
        print(ucode, 'unit', ul, 'err on that code', round((e_[m_, :32] + u_empty[:, :32] - r1[m_]).abs().mean().item(), 5))
```