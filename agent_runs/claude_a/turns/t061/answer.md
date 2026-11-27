Still not exact for glucose_finger. The error is 0.03 to 0.06 and rises only slightly with the value. The lab normalisation and the extra unused rows do not help. So glucose_finger atoms must carry some other term that MAP and glucose_lab atoms lack.

Next I will search more widely, on the 50 glucose_finger single-atom bundles only, over:
- the unit row (including none);
- a rate_unit row;
- the numeric dose and rate columns set from the value;
- the dose or rate bin tables indexed by the value bin.

If one combination is exact, I have the last measurement convention. If none is, the difference is in something I cannot see from single atoms, and I will treat glucose_finger's 0.05 error as tolerable for now.

```python
res = []
vbg = torch.tensor(b_std); zg = torch.tensor(z_std, dtype=torch.float32)
num_opts = {'none': [], 'c2z': [(2, 'z')], 'c3one': [(3, 1)], 'c4z': [(4, 'z')], 'c5one': [(5, 1)], 'c2z_c3': [(2, 'z'), (3, 1)], 'c4z_c5': [(4, 'z'), (5, 1)], 'c2z_c3_c4z_c5': [(2, 'z'), (3, 1), (4, 'z'), (5, 1)]}
size_g = lin(torch.full((len(subg), 1), np.log(2.0)), 'encode.size')
for un_ in [0, 1, 2, 3, 4, 5]:
    for ru_ in [0, 1, 2, 3]:
        for nk, nlist in num_opts.items():
            nf = torch.zeros(len(subg), 6); nf[:, 0] = zg; nf[:, 1] = 1
            for col, val in nlist: nf[:, col] = zg if val == 'z' else 1.0
            for bx in ['none', 'dose', 'rate', 'both']:
                av_ = (P['embed.code.weight'][cg] + P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + P['embed.kq.weight'][0]
                       + P['embed.categorical.unit.weight'][un_] + P['embed.categorical.rate_unit.weight'][ru_] + P['embed.bins.value.weight'][vbg] + lin(nf, 'embed.numeric'))
                if bx in ('dose', 'both'): av_ = av_ + P['embed.bins.dose.weight'][vbg]
                if bx in ('rate', 'both'): av_ = av_ + P['embed.bins.rate.weight'][vbg]
                e_ = ln(mlp(mlp(av_, 'encode.mlp1', F.gelu) + size_g, 'encode.mlp2', F.gelu), 'encode.norm')
                res.append(((e_[:, :32] + u_empty[:, :32] - rg).abs().mean().item(), un_, ru_, nk, bx))
res.sort(key=lambda r: r[0])
for r in res[:6]: print(r)
```