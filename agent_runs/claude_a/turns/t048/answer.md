The z posterior still matches under no combination: the best error is 0.054 against a target scale of 0.106. The undecayed context keeps doing better, and the residual-posterior form does not help. So I am still missing a basic convention.

To find it, I will search more widely but on the simplest bundles only: those with a single measurement atom, mostly hour −1. That makes each trial cheap enough to try every combination of:
- numeric column placement;
- unit row;
- whether kind, event kind, kq and the value-bin terms are added;
- ctx with or without decay.

If some combination drives the error near zero, the atom embedding is identified. If none does, the fault is outside the atom sum, in ctx or in how z_post is used, and I will drop the core rebuild for good.

```python
n1_rows = [r for r in only_meas if SB.n_atoms.values[r] == 1]
print('single-atom meas bundles', len(n1_rows))
sub1 = AT[AT.row.isin(n1_rows)].copy(); sub1 = sub1.set_index('row').loc[n1_rows].reset_index()
z1 = ZP[torch.tensor(n1_rows)]
pv1 = torch.tensor(SB.prev.values[n1_rows].astype(np.int64)); dt1 = torch.tensor(SB.dts.values[n1_rows], dtype=torch.float32)
g1 = mlp((torch.log(dt1)/10)[:, None], 'gap', F.gelu)
ctx1 = {'d': OUT[pv1] + ln((HS[pv1]*torch.exp(-dt1[:, None, None]*torch.exp(P['core.flow.a']))).mean(1), 'core.out_norm') + g1,
        'nd': OUT[pv1] + ln(HS[pv1].mean(1), 'core.out_norm') + g1}
c1 = torch.tensor(sub1.event.map(cid).values); zv1 = np.zeros(len(sub1)); vb1 = np.zeros(len(sub1), np.int64)
for ev in codes_names:
    m = (sub1.event == ev).values
    if m.any(): zv1[m] = zsc(ev, 'value', sub1.value.values[m]); vb1[m] = fbin(ev, 'value', sub1.value.values[m])
zv1t = torch.tensor(zv1, dtype=torch.float32)
isg = torch.tensor(sub1.event.str.startswith('glucose').values)
size1 = lin(torch.full((len(sub1), 1), np.log(2.0)), 'encode.size')
res = []
for cz, ch in itertools.permutations(range(6), 2):
    nf = torch.zeros(len(sub1), 6); nf[:, cz] = zv1t; nf[:, ch] = 1
    numt = lin(nf, 'embed.numeric')
    for ug, um in [(None, None)] + list(itertools.permutations([2, 3, 4, 5], 2)):
        ut = 0 if ug is None else P['embed.categorical.unit.weight'][torch.where(isg, torch.tensor(ug), torch.tensor(um))]
        for ak, ae, aq, avb in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
            av = P['embed.code.weight'][c1] + ak*P['embed.kind.weight'][0] + ae*P['embed.ekind.weight'][1] + aq*P['embed.kq.weight'][0] + ut + avb*P['embed.bins.value.weight'][torch.tensor(vb1)] + numt
            e1 = ln(mlp(mlp(av, 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm')
            for cn in ['d', 'nd']:
                po = mlp(torch.cat([ctx1[cn], e1, torch.zeros_like(e1)], 1), 'z_post', F.gelu)[:, :32]
                res.append(((po - z1).abs().mean().item(), cz, ch, ug, um, ak, ae, aq, avb, cn))
res.sort(key=lambda r: r[0])
for r in res[:8]: print(tuple(round(x, 4) if isinstance(x, float) else x for x in r))
print('z1 scale', z1.abs().mean().item())
```