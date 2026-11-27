The atom embedding is nearly solved. MAP (arterial and cuff) reproduces exactly with unit row 2, and glucose_lab is exact with unit row 5. Only glucose_finger is still off, at 0.047 with row 3, so it must use a different unit row.

Next I will try rows 2 to 5 for glucose_finger. Then I will re-check the multi-atom bundles with the corrected units, comparing 1/√n with 1/n aggregation. One row should make glucose_finger exact. With that, the multi-atom error should drop to about zero under the documented 1/√n aggregation.

```python
for ul in [2, 3, 4, 5]:
    m_ = (sub1.event == 'glucose_finger').values
    umap_ = {'glucose_finger': ul, 'glucose_lab': 5, 'map_arterial': 2, 'map_cuff': 2}
    ut_ = P['embed.categorical.unit.weight'][torch.tensor(sub1.event.map(umap_).values)]
    nf = torch.zeros(len(sub1), 6); nf[:, 0] = zv1t; nf[:, 1] = 1
    av_ = P['embed.code.weight'][c1] + P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + P['embed.kq.weight'][0] + ut_ + P['embed.bins.value.weight'][torch.tensor(vb1)] + lin(nf, 'embed.numeric')
    e_ = ln(mlp(mlp(av_, 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm')
    print('glucose_finger unit', ul, round((e_[m_, :32] + u_empty[:, :32] - r1[m_]).abs().mean().item(), 6))
UNIT = {'glucose_finger': None, 'glucose_lab': 5, 'map_arterial': 2, 'map_cuff': 2}
```