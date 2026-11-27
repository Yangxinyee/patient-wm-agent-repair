Found it. A bundle with no actions does not get u = 0. It gets the encoding of an empty action set, LN(mlp2(size(0))). Adding that cut the error from 0.40 to 0.013. The same surely holds for e_k in action-only bundles, and it explains why every z_post check failed.

Next I will redo the convention search with this offset included, then check bundles with several measurement atoms to confirm the 1/√n and size(log(1 + n)) terms. I expect an exact match, near 1e-5, for one convention. If the best stays near 0.01, a smaller detail is still wrong.

```python
e_empty = ln(mlp(lin(torch.zeros(1, 1), 'encode.size'), 'encode.mlp2', F.gelu), 'encode.norm')
res = []
for cz, ch in itertools.permutations(range(6), 2):
    nf = torch.zeros(len(sub1), 6); nf[:, cz] = zv1t; nf[:, ch] = 1
    numt = lin(nf, 'embed.numeric')
    for ug, um in [(None, None)] + list(itertools.permutations([2, 3, 4, 5], 2)):
        ut = 0 if ug is None else P['embed.categorical.unit.weight'][torch.where(isg, torch.tensor(ug), torch.tensor(um))]
        for ak, ae, aq, avb in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
            av = P['embed.code.weight'][c1] + ak*P['embed.kind.weight'][0] + ae*P['embed.ekind.weight'][1] + aq*P['embed.kq.weight'][0] + ut + avb*P['embed.bins.value.weight'][torch.tensor(vb1)] + numt
            e1_ = ln(mlp(mlp(av, 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm')
            res.append(((e1_[:, :32] + u_empty[:, :32] - r1).abs().mean().item(), cz, ch, ug, um, ak, ae, aq, avb))
res.sort(key=lambda r: r[0])
for r in res[:5]: print(tuple(round(x, 6) if isinstance(x, float) else x for x in r))
_, cz, ch, ug, um, ak, ae, aq, avb = res[0]
MEAS_CFG = dict(cz=cz, ch=ch, ug=ug, um=um, ak=ak, ae=ae, aq=aq, avb=avb)
# multi-atom measurement bundles: check sqrt(n) & size
def enc_meas2(subA_, rows_, cfg, A=F.gelu):
    rm_ = {r: i for i, r in enumerate(rows_)}; g_ = torch.tensor(subA_.row.map(rm_).values)
    c_ = torch.tensor(subA_.event.map(cid).values); v_ = subA_.value.values; zv_ = np.zeros(len(subA_)); vb_ = np.zeros(len(subA_), np.int64)
    for ev in codes_names:
        m = (subA_.event == ev).values
        if m.any(): zv_[m] = zsc(ev, 'value', v_[m]); vb_[m] = fbin(ev, 'value', v_[m])
    nf = torch.zeros(len(subA_), 6); nf[:, cfg['cz']] = torch.tensor(zv_, dtype=torch.float32); nf[:, cfg['ch']] = 1
    isg_ = torch.tensor(subA_.event.str.startswith('glucose').values)
    ut = 0 if cfg['ug'] is None else P['embed.categorical.unit.weight'][torch.where(isg_, torch.tensor(cfg['ug']), torch.tensor(cfg['um']))]
    av = (P['embed.code.weight'][c_] + cfg['ak']*P['embed.kind.weight'][0] + cfg['ae']*P['embed.ekind.weight'][1] + cfg['aq']*P['embed.kq.weight'][0]
          + ut + cfg['avb']*P['embed.bins.value.weight'][torch.tensor(vb_)] + lin(nf, 'embed.numeric'))
    return ds_act(av, g_, len(rows_), 'encode', A)
rows_m = [r for r in only_meas if SB.n_atoms.values[r] > 1][:800]
subM = AT[AT.row.isin(rows_m)].copy()
eM = enc_meas2(subM, rows_m, MEAS_CFG)
print('multi-atom meas err', (eM[:, :32] + u_empty[:, :32] - REV[torch.tensor(rows_m)]).abs().mean().item(), 'scale', REV[torch.tensor(rows_m)].abs().mean().item())
```