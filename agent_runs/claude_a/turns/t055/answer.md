The reveal ran. Next I will drop the diagnostic edit at once, so it cannot leak into later passes, and then load the revealed values. I will check two things:
- whether bundles with identical content (same code and value, single atom) give identical revealed vectors, which would confirm the values do not depend on the state;
- whether, for measurement-only bundles, I can reproduce the revealed first 32 dims of e_k by searching the atom-embedding conventions (numeric columns, unit row, kind / event kind / kq / value-bin terms).

An exact match, around 1e-5, would identify the measurement encoder. If even identical bundles disagree, the reveal does not show pure encoder outputs and I must rethink.

```python
for n_ in ['z_post.0.bias', 'z_post.0.weight', 'z_post.2.bias', 'z_post.2.weight']: drop_edit(n_)
print('edited now', edited())
S4 = np.load('/work/fwd_4/states_edited.npz'); REV = torch.tensor(S4['x'][:, 960:992])
assert (S4['stay_id'] == sids_S).all() and np.allclose(S4['hour'], hrs_S)
# sanity: identical single-atom bundles -> identical revealed
sub1['key'] = sub1.event + '_' + sub1.value.astype(str)
dup = sub1.groupby('key').row.apply(list)
dup = dup[dup.apply(len) > 1]
print('dup groups', len(dup), 'max within-group spread', max((REV[torch.tensor(v)] - REV[torch.tensor(v)].mean(0)).abs().max().item() for v in dup.values) if len(dup) else None)
print('revealed scale (meas n1)', REV[torch.tensor(n1_rows)].abs().mean().item())
r1 = REV[torch.tensor(n1_rows)]
res = []
for cz, ch in itertools.permutations(range(6), 2):
    nf = torch.zeros(len(sub1), 6); nf[:, cz] = zv1t; nf[:, ch] = 1
    numt = lin(nf, 'embed.numeric')
    for ug, um in [(None, None)] + list(itertools.permutations([2, 3, 4, 5], 2)):
        ut = 0 if ug is None else P['embed.categorical.unit.weight'][torch.where(isg, torch.tensor(ug), torch.tensor(um))]
        for ak, ae, aq, avb in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
            av = P['embed.code.weight'][c1] + ak*P['embed.kind.weight'][0] + ae*P['embed.ekind.weight'][1] + aq*P['embed.kq.weight'][0] + ut + avb*P['embed.bins.value.weight'][torch.tensor(vb1)] + numt
            e1_ = ln(mlp(mlp(av, 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm')
            res.append(((e1_[:, :32] - r1).abs().mean().item(), cz, ch, ug, um, ak, ae, aq, avb))
res.sort(key=lambda r: r[0])
for r in res[:6]: print(tuple(round(x, 5) if isinstance(x, float) else x for x in r))
```