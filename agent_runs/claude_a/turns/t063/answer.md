The measurement encoder is now exact for every measurement-only bundle (error 5e-7), and the aggregation is confirmed as 1/√n. The action bundles hold four kinds of atom in equal pairs: insulin start and END (1,825 each), and norepinephrine start and END (2,394 each).

Next I will identify the drug-start atoms on action bundles with a single start atom. Their revealed value is u_k plus the empty-measurement encoding. I will search the rate-unit row, the unit row, and whether the dose and rate bin rows are added, with dose and rate in numeric columns 2 to 5 as documented. One exact combination per drug will fix the start-atom embedding. After that I can work out the END atoms from start-plus-END bundles.

```python
e_emptyM = ln(mlp(lin(torch.zeros(1, 1), 'encode.size'), 'encode.mlp2', F.gelu), 'encode.norm')
nX = subX.groupby('row').size()
single_start = [r for r in act_only if nX[r] == 1 and (subX[subX.row == r].kind.values[0] == 0)]
subS = AT[AT.row.isin(single_start)].copy().set_index('row').loc[single_start].reset_index()
print('single-start bundles', len(single_start), subS.event.value_counts().to_dict())
rS = REV[torch.tensor(single_start)]
def drug_feats(sub_):
    zd = np.zeros(len(sub_)); zr = np.zeros(len(sub_)); db = np.zeros(len(sub_), np.int64); rb = np.zeros(len(sub_), np.int64)
    for dn_ in ['insulin', 'norepinephrine']:
        m = (sub_.event == dn_).values
        if not m.any(): continue
        rr = sub_.rate.values[m]; dd = rr * (1.0 if dn_ == 'insulin' else 4.8)
        zr[m] = zsc(dn_, 'rate', rr); zd[m] = zsc(dn_, 'dose', dd); rb[m] = fbin(dn_, 'rate', rr); db[m] = fbin(dn_, 'dose', dd)
    return zd, zr, db, rb
zdS, zrS, dbS, rbS = drug_feats(subS)
cS = torch.tensor(subS.event.map(cid).values); isins = torch.tensor((subS.event == 'insulin').values)
sizeS = lin(torch.full((len(subS), 1), np.log(2.0)), 'encode_action.size')
res = []
for ru_i, ru_n in itertools.product(range(4), range(4)):
    for du_i, du_n in itertools.product(range(6), range(6)):
        if du_i not in (0, 3, 4, 5) or du_n not in (0, 3, 4, 5): continue
        for adb, arb in itertools.product([0, 1], [0, 1]):
            nf = torch.zeros(len(subS), 6); nf[:, 2] = torch.tensor(zdS, dtype=torch.float32); nf[:, 3] = 1; nf[:, 4] = torch.tensor(zrS, dtype=torch.float32); nf[:, 5] = 1
            av_ = (P['embed.code.weight'][cS] + P['embed.kind.weight'][0] + P['embed.ekind.weight'][2] + P['embed.kq.weight'][0]
                   + P['embed.categorical.rate_unit.weight'][torch.where(isins, torch.tensor(ru_i), torch.tensor(ru_n))]
                   + P['embed.categorical.unit.weight'][torch.where(isins, torch.tensor(du_i), torch.tensor(du_n))]
                   + adb*P['embed.bins.dose.weight'][torch.tensor(dbS)] + arb*P['embed.bins.rate.weight'][torch.tensor(rbS)] + lin(nf, 'embed.numeric'))
            u_ = ln(mlp(mlp(av_, 'encode_action.mlp1', F.gelu) + sizeS, 'encode_action.mlp2', F.gelu), 'encode_action.norm')
            er_ = (u_[:, :32] + e_emptyM[:, :32] - rS).abs().mean(1)
            res.append((er_[isins].mean().item() if isins.any() else 0, er_[~isins].mean().item() if (~isins).any() else 0, ru_i, ru_n, du_i, du_n, adb, arb))
res.sort(key=lambda r: r[0] + r[1])
for r in res[:6]: print(tuple(round(x, 6) if isinstance(x, float) else x for x in r))
print('best insulin', sorted(res, key=lambda r: r[0])[0][:1], 'best ne', sorted(res, key=lambda r: r[1])[0][:2])
```