The END atoms are settled. They go through the measurement encoder, as world events, and carry only code, kind 2, event kind 2, kq 0 and the numeric bias, with no rate, dose, units or bins. With that the action bundles reproduce to 0.00018.

So the encoders split the bundle like this:
- e_k encodes the measurements, visits and END atoms;
- u_k encodes only the drug-start atoms with their doses.

The leftover error is small and probably sits in the norepinephrine start atoms, where a dose or rate falls exactly on a bin edge.

Next I will find which bundles carry the residual and test the opposite bin-edge rule for the rate and dose bins. I will also check the visit bundles: hospital admission at −2 and ICU admission plus readings at 0. If the residual falls to 1e-6, the encoders are exactly reproduced for every bundle type. That opens the way to the z_post and core checks.

```python
def drug_feats3(sub_, side='left'):
    zd = np.zeros(len(sub_)); zr = np.zeros(len(sub_)); db = np.zeros(len(sub_), np.int64); rb = np.zeros(len(sub_), np.int64)
    for dn_ in ['insulin', 'norepinephrine']:
        m = (sub_.event == dn_).values
        if not m.any(): continue
        rr = sub_.rate.values[m]; dd = sub_.dose.values[m]
        zr[m] = zsc(dn_, 'rate', rr); zd[m] = zsc(dn_, 'dose', dd)
        rb[m] = 1 + np.searchsorted(np.array(num[(dn_, 'rate')]['fine_bin_edges']), rr, side=side)
        db[m] = 1 + np.searchsorted(np.array(num[(dn_, 'dose')]['fine_bin_edges']), dd, side=side)
    return zd, zr, db, rb
def start_atoms(sub_, side='left'):
    zd, zr, db, rb = drug_feats3(sub_, side)
    c_ = torch.tensor(sub_.event.map(cid).values); ru = torch.tensor(sub_.event.map(RU).values); du = torch.tensor(sub_.event.map(DU).values)
    nf = torch.zeros(len(sub_), 6); nf[:, 2] = torch.tensor(zd, dtype=torch.float32); nf[:, 3] = 1; nf[:, 4] = torch.tensor(zr, dtype=torch.float32); nf[:, 5] = 1
    return (P['embed.code.weight'][c_] + P['embed.ekind.weight'][2] + P['embed.kq.weight'][0] + P['embed.kind.weight'][0]
            + P['embed.categorical.rate_unit.weight'][ru] + P['embed.categorical.unit.weight'][du]
            + P['embed.bins.dose.weight'][torch.tensor(db)] + P['embed.bins.rate.weight'][torch.tensor(rb)] + lin(nf, 'embed.numeric'))
END_CFG = dict(ru=0, du=0, db=0, rb=0, num=None)
e_en = encode_set(end_atoms(subEn, END_CFG), torch.tensor(subEn.row.map(rmX).values), len(rows_x), 'encode')
for side in ['left', 'right']:
    u_s = encode_set(start_atoms(subSt, side), torch.tensor(subSt.row.map(rmX).values), len(rows_x), 'encode_action')
    errb = (u_s[:, :32] + e_en[:, :32] - rX).abs().mean(1)
    print(side, 'mean err', errb.mean().item(), 'n bundles err>1e-4', (errb > 1e-4).sum().item())
    if side == 'left':
        bad = [rows_x[i] for i in torch.nonzero(errb > 1e-4).flatten().tolist()]
        print(AT[AT.row.isin(bad[:3])][['row', 'event', 'kind', 'rate', 'dose']].to_string())
# visit bundles
vis_list = sorted(vis_rows)
print('visit bundles', len(vis_list), AT[AT.row.isin(vis_list[:4])][['row', 'hour', 'event', 'kind', 'value']].to_string())
```