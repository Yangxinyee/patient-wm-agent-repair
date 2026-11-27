The insulin start atom is exact: rate-unit row 2, unit row 3, with both the dose-bin and rate-bin rows added. Norepinephrine start is almost exact with rate-unit row 3 and unit row 4. Its tiny residual (0.0004) probably comes from computing the dose as rate × 4.8, which can land on the other side of a repeated bin edge. Using the recorded dose column should remove it.

Next I will switch to the recorded doses and identify the END atoms, using all action-only bundles together with the now-known start atoms. For END atoms I will search whether they carry the rate-unit and unit rows, the bin rows, and the numeric dose and rate of the record that ended. One option should make the whole action encoder exact. That would give me u_k for any dose, which is what I need to see how doubled doses enter the core.

```python
dose_map = {(a, round(b, 4), c): d for a, b, c, d in zip(tr.stay_id, tr.hour, tr.event, tr.dose) if c in ('insulin', 'norepinephrine')}
AT['dose'] = np.nan
mstart = AT.event.isin(['insulin', 'norepinephrine'])
# for END atoms, the record started 1h earlier
AT.loc[mstart, 'dose'] = [dose_map.get((a, round(h - (0 if k == 0 else 1.0), 4), e), np.nan) for a, h, e, k in zip(AT.stay_id[mstart], AT.hour[mstart], AT.event[mstart], AT.kind[mstart])]
print('missing dose', AT.loc[mstart, 'dose'].isna().sum())
def drug_feats2(sub_):
    zd = np.zeros(len(sub_)); zr = np.zeros(len(sub_)); db = np.zeros(len(sub_), np.int64); rb = np.zeros(len(sub_), np.int64)
    for dn_ in ['insulin', 'norepinephrine']:
        m = (sub_.event == dn_).values
        if not m.any(): continue
        rr = sub_.rate.values[m]; dd = sub_.dose.values[m]
        zr[m] = zsc(dn_, 'rate', rr); zd[m] = zsc(dn_, 'dose', dd); rb[m] = fbin(dn_, 'rate', rr); db[m] = fbin(dn_, 'dose', dd)
    return zd, zr, db, rb
RU = {'insulin': 2, 'norepinephrine': 3}; DU = {'insulin': 3, 'norepinephrine': 4}
def action_atoms(sub_, end_cfg):
    zd, zr, db, rb = drug_feats2(sub_)
    isend = torch.tensor((sub_.kind == 2).values); st = ~isend
    c_ = torch.tensor(sub_.event.map(cid).values)
    ru = torch.tensor(sub_.event.map(RU).values); du = torch.tensor(sub_.event.map(DU).values)
    nf = torch.zeros(len(sub_), 6); nf[:, 2] = torch.tensor(zd, dtype=torch.float32); nf[:, 3] = 1; nf[:, 4] = torch.tensor(zr, dtype=torch.float32); nf[:, 5] = 1
    av = P['embed.code.weight'][c_] + P['embed.ekind.weight'][2] + P['embed.kq.weight'][0] + torch.where(isend[:, None], P['embed.kind.weight'][2], P['embed.kind.weight'][0])
    sw = st.float()[:, None]; ew = isend.float()[:, None]
    av = av + sw*(P['embed.categorical.rate_unit.weight'][ru] + P['embed.categorical.unit.weight'][du] + P['embed.bins.dose.weight'][torch.tensor(db)] + P['embed.bins.rate.weight'][torch.tensor(rb)] + lin(nf, 'embed.numeric'))
    nf_e = nf.clone()
    if not end_cfg['num']: nf_e[:] = 0
    elif end_cfg['num'] == 'has': nf_e[:, 2] = 0; nf_e[:, 4] = 0
    av = av + ew*(end_cfg['ru']*P['embed.categorical.rate_unit.weight'][ru] + end_cfg['du']*P['embed.categorical.unit.weight'][du]
                  + end_cfg['db']*P['embed.bins.dose.weight'][torch.tensor(db)] + end_cfg['rb']*P['embed.bins.rate.weight'][torch.tensor(rb)]
                  + F.linear(nf_e, P['embed.numeric.weight']) + P['embed.numeric.bias'])
    return av
rows_x = act_only[:700]; subXX = AT[AT.row.isin(rows_x)].copy(); rmX = {r: i for i, r in enumerate(rows_x)}; gX = torch.tensor(subXX.row.map(rmX).values)
rX = REV[torch.tensor(rows_x)]
res = []
for ru_, du_, db_, rb_, num_ in itertools.product([0, 1], [0, 1], [0, 1], [0, 1], [None, 'has', 'full']):
    cfg = dict(ru=ru_, du=du_, db=db_, rb=rb_, num=num_)
    u_ = ds_act(action_atoms(subXX, cfg), gX, len(rows_x), 'encode_action', F.gelu)
    res.append(((u_[:, :32] + e_emptyM[:, :32] - rX).abs().mean().item(), cfg))
res.sort(key=lambda r: r[0])
for r in res[:5]: print(round(r[0], 7), r[1])
```