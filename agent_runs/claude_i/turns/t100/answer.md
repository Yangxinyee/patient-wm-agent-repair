Every measurement bundle now matches exactly, so the measurement encoder and ctx are settled: glucose_finger unit 0, glucose_lab 5, MAP 2, kq 0, encoded empty sets. The 'none' rows are the stay-end bundles of admission END atoms, which I haven't modelled and which don't matter for predictions. The remaining errors are all in drug bundles. Next I'll search the drug conventions on drug-only bundles:
- the unit index for insulin and norepinephrine (0-5 each);
- the rate-unit indices;
- whether END atoms carry rate and dose.

I'll encode only the drug atoms to save time. An exact zero for one setting would complete the encoder.

```python
Ad = A3[A3.code.isin(['insulin','norepinephrine'])].copy()
do = (prev_ok & (encC['ne'] == 0)).numpy()
def u_only(cfg, act='gelu'):
    V = atoms_embed(Ad, cfg); bi = torch.tensor(Ad.bi.values)
    H = mlp_(V, 'encode_action.mlp1', act)
    S = torch.zeros(NB, 192, dtype=torch.float64).index_add_(0, bi, H)
    n = torch.zeros(NB, dtype=torch.float64).index_add_(0, bi, torch.ones(len(Ad), dtype=torch.float64))
    S = S / torch.sqrt(n.clamp(min=1))[:, None] + torch.log1p(n)[:, None] * Wt('encode_action.size.weight')[:, 0] + Wt('encode_action.size.bias')
    return LN(mlp_(S, 'encode_action.mlp2', act), (192,), Wt('encode_action.norm.weight'), Wt('encode_action.norm.bias'), eps=1e-5)
res3 = []
for ui in range(6):
    for un in range(6):
        for rr_ in [(2,3),(3,2)]:
            for ea in [True, False]:
                cfg = {**cfgC, 'unit': {**cfgC['unit'], 'insulin': ui, 'norepinephrine': un}, 'runit': {'insulin': rr_[0], 'norepinephrine': rr_[1]}, 'end_attrs': ea}
                u_ = u_only(cfg)
                zp = mlp_(torch.cat([ctxF, encC['e'], u_], 1), 'z_post', 'gelu')[:, :32]
                er_ = (zp - zm3).abs().mean(1).numpy()
                res3.append((round(er_[do].mean(), 5), ui, un, rr_, ea))
res3.sort(); print(res3[:6])
```