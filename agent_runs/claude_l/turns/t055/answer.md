The first-bundle search failed to match: the best maximum error is 0.87, against z values of about ±2.8. The best gap input (−0.125) does not correspond to any meaningful time gap, which suggests the initial-state assumption is wrong, rather than the activations.

To separate the two questions, I will test the second bundle of each stay instead. There ctx comes entirely from known states: the true `out` and h after bundle 1, decayed over the known dt, plus gap of dt. Only the encoders, `z_post`, `gap` and the gap-input convention remain unknown.

If one combination matches to about 1e-4 across 100 stays, the encoders and `z_post` are settled and only the start-of-stay convention is left. If nothing matches, the encoder or the atom embedding itself is wrong.

```python
sec_idx = first + 1
dts = (hrv[sec_idx] - hrv[first]) * 3600.0
fa64 = torch.exp(T64('core.flow.a'))
ctx_base = outS[first].double() + F.layer_norm((hS[first].double().mean(1)) * torch.exp(-torch.tensor(dts)[:, None] * fa64[None]), (192,), T64('core.out_norm.weight'), onb, 1e-5)
z1_true = zS[sec_idx].double()
atoms1 = [dict(BUN[s_])[hrv[i_]] if False else None for s_, i_ in zip(batch1, sec_idx)]
atoms1 = []
for s_, i_ in zip(batch1, sec_idx):
    bl_ = BUN[s_]; tt_ = hrv[i_]; atoms1.append([b_[1] for b_ in bl_ if abs(b_[0] - tt_) < 1e-3][0])
print('n atoms in second bundles:', np.bincount([len(a) for a in atoms1]), 'any action:', any(a_['act'] for aa in atoms1 for a_ in aa))
gvar = {'log_s/10': np.log(dts)/10, 'log1p_s/10': np.log1p(dts)/10, 'log_h/10': np.log(dts/3600)/10, 'log1p_h/10': np.log1p(dts/3600)/10, 'log_s': np.log(dts)}
res2 = []
for a1n, a2n in itertools.product(ACT, ACT):
    for kq in [True]:
        E1 = torch.stack([setenc(aa, 'encode', ACT[a1n], ACT[a2n], kq) for aa in atoms1])
        for gan in ACT:
            for gvn, gv_ in gvar.items():
                ctx_ = ctx_base + mlp2l(torch.tensor(gv_)[:, None], 'gap', ACT[gan])
                inp_ = torch.cat([ctx_, E1, torch.zeros_like(E1)], 1)
                for zn in ACT:
                    zm = mlp2l(inp_, 'z_post', ACT[zn])[:, :32]
                    er = (zm - z1_true).abs()
                    res2.append((er.mean().item(), er.max().item(), a1n, a2n, gan, gvn, zn))
res2.sort(key=lambda r: r[0])
for r_ in res2[:10]: print(np.round(r_[:2], 5), r_[2:])
```