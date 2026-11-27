The bundles match perfectly: 5,369 of 5,369, with every atom count right. The z posterior check failed, though: an error of 0.088 against a target size of 0.106 means my e_k is wrong. The architecture text already gave the `last_numeric` layout wrongly, so its column order for `embed.numeric` (z_value, has_value, …) may be wrong too. Whether `kq` and `kind` rows are added is also uncertain.

Next, on a subset of 400 measurement-only bundles, I will search:
- which two numeric columns hold z_value and has_value;
- whether kq row 0 and kind row 0 are added;
- two candidate unit maps.

As a sanity check on the target, I will also see how closely the stored z resembles the z prior mean. A combination with near-zero error would confirm the encoder. If every combination stays around 0.08, the mismatch lies elsewhere, perhaps in the z_post input order or in what the stored z actually is.

```python
sub_rows = only_meas[:400]
ctxs = ctx_rows(sub_rows)
subA2 = AT[AT.row.isin(sub_rows)].copy(); rmap2 = {r: i for i, r in enumerate(sub_rows)}
grp2 = torch.tensor(subA2.row.map(rmap2).values); zt2 = ZP[torch.tensor(sub_rows)]
pri = mlp(ctxs, 'z_prior', F.gelu)
print('zt vs prior mean: mae', (pri[:, :32] - zt2).abs().mean().item(), 'corr', np.corrcoef(pri[:, :32].flatten().numpy(), zt2.flatten().numpy())[0, 1].round(3))
c2 = torch.tensor(subA2.event.map(cid).values)
v2 = subA2.value.values; zv2 = np.zeros(len(subA2)); vb2 = np.zeros(len(subA2), np.int64)
for ev in codes_names:
    m = (subA2.event == ev).values
    if m.any(): zv2[m] = zsc(ev, 'value', v2[m]); vb2[m] = fbin(ev, 'value', v2[m])
zv2t = torch.tensor(zv2, dtype=torch.float32); vb2t = torch.tensor(vb2)
res = []
for (ug, um) in [(3, 2), (4, 2), (2, 3), (5, 4)]:
    u2 = torch.tensor(subA2.event.map({'glucose_finger': ug, 'glucose_lab': ug, 'map_arterial': um, 'map_cuff': um}).values)
    basev = P['embed.code.weight'][c2] + P['embed.ekind.weight'][1] + P['embed.categorical.unit.weight'][u2] + P['embed.bins.value.weight'][vb2t] + P['embed.numeric.bias']
    for cz, ch in itertools.permutations(range(6), 2):
        numf = torch.zeros(len(subA2), 6); numf[:, cz] = zv2t; numf[:, ch] = 1
        for add_kq, add_kind in itertools.product([0, 1], [0, 1]):
            av = basev + numf @ P['embed.numeric.weight'].T + add_kq*P['embed.kq.weight'][0] + add_kind*P['embed.kind.weight'][0]
            e_ = deepsets(av, grp2, len(sub_rows), 'encode')
            zp_ = mlp(torch.cat([ctxs, e_, torch.zeros_like(e_)], 1), 'z_post', F.gelu)[:, :32]
            res.append(((zp_ - zt2).abs().mean().item(), ug, um, cz, ch, add_kq, add_kind))
res.sort(); print(res[:8])
```