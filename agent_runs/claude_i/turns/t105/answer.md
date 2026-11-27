Insulin unit 3, norepinephrine unit 4 and rate units (2, 3) bring the single-START error down to 0.00014, essentially exact. Next I'll test this setting on all drug bundles, which include END atoms, with END atoms carrying or not carrying rate and dose. I'll report the error by bundle composition. Errors near 1e-4 everywhere would confirm the full encoder; a larger error on bundles with END atoms would show how END atoms are encoded.

```python
cfgE = {**cfgD, 'unit': {**cfgD['unit'], 'insulin': 3, 'norepinephrine': 4}, 'runit': {'insulin': 2, 'norepinephrine': 3}}
for ea in [False, True]:
    cfg = {**cfgE, 'end_attrs': ea}
    u_ = u_only_x(cfg, {'kq': 1, 'action': 1})
    zp = mlp_(torch.cat([ctxF, encC['e'], u_], 1), 'z_post', 'gelu')[:, :32]
    er_ = (zp - zm3).abs().mean(1).numpy()
    dfE = pd.DataFrame({'comp': comp, 'err': er_, 'ok': prev_ok.numpy()}); dfE = dfE[dfE.ok & (dfE.comp != 'none')]
    print('end_attrs', ea, 'overall', round(dfE.err.mean(), 6), dfE.groupby('comp').err.mean().sort_values(ascending=False).head(5).round(5).to_dict())
```