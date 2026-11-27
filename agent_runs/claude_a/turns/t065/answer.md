The END atoms fail badly, with an error of 0.79 under every option. Treating them as actions must be wrong. They are probably processed as world events by the measurement encoder: the record ending is something that happens, not a clinical choice. In that case e_k is the encoding of the END atoms, and u_k covers only the start atoms.

Next I will test that split on all action-only bundles:
- e = encode(END atoms);
- u = encode_action(start atoms);
- u = the empty-action encoding when a bundle holds only END atoms.

I will search the END-atom conventions under this split. If the error drops to about zero, I know how every bundle is encoded, and the action encoder sees only the start atoms with their doses.

```python
def end_atoms(sub_, cfg):
    zd, zr, db, rb = drug_feats2(sub_)
    c_ = torch.tensor(sub_.event.map(cid).values)
    ru = torch.tensor(sub_.event.map(RU).values); du = torch.tensor(sub_.event.map(DU).values)
    nf = torch.zeros(len(sub_), 6)
    if cfg['num'] in ('has', 'full'): nf[:, 3] = 1; nf[:, 5] = 1
    if cfg['num'] == 'full': nf[:, 2] = torch.tensor(zd, dtype=torch.float32); nf[:, 4] = torch.tensor(zr, dtype=torch.float32)
    av = (P['embed.code.weight'][c_] + P['embed.ekind.weight'][2] + P['embed.kq.weight'][0] + P['embed.kind.weight'][2] + lin(nf, 'embed.numeric')
          + cfg['ru']*P['embed.categorical.rate_unit.weight'][ru] + cfg['du']*P['embed.categorical.unit.weight'][du]
          + cfg['db']*P['embed.bins.dose.weight'][torch.tensor(db)] + cfg['rb']*P['embed.bins.rate.weight'][torch.tensor(rb)])
    return av
START_CFG = dict(ru=0, du=0, db=0, rb=0, num=None)
def encode_set(av, rows_idx, ngrp, pre):
    out = ln(mlp(lin(torch.zeros(1, 1), pre + '.size'), pre + '.mlp2', F.gelu), pre + '.norm').repeat(ngrp, 1)
    if len(av) == 0: return out
    h1 = mlp(av, pre + '.mlp1', F.gelu); s = torch.zeros(ngrp, 192).index_add_(0, rows_idx, h1); n = torch.zeros(ngrp).index_add_(0, rows_idx, torch.ones(len(rows_idx)))
    s2 = s / torch.sqrt(n.clamp(min=1))[:, None] + lin(torch.log1p(n)[:, None], pre + '.size')
    enc = ln(mlp(s2, pre + '.mlp2', F.gelu), pre + '.norm')
    return torch.where((n > 0)[:, None], enc, out)
subSt = subXX[subXX.kind == 0]; subEn = subXX[subXX.kind == 2]
u_st = encode_set(action_atoms(subSt, START_CFG), torch.tensor(subSt.row.map(rmX).values), len(rows_x), 'encode_action')
res = []
for ru_, du_, db_, rb_, num_ in itertools.product([0, 1], [0, 1], [0, 1], [0, 1], [None, 'has', 'full']):
    cfg = dict(ru=ru_, du=du_, db=db_, rb=rb_, num=num_)
    e_en = encode_set(end_atoms(subEn, cfg), torch.tensor(subEn.row.map(rmX).values), len(rows_x), 'encode')
    res.append(((u_st[:, :32] + e_en[:, :32] - rX).abs().mean().item(), cfg))
res.sort(key=lambda r: r[0])
for r in res[:5]: print(round(r[0], 7), r[1])
```