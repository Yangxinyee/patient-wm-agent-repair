Giving drug atoms kq index 1 plus the action-table row 1 cuts the single-START error from 0.10 to 0.0014. Rows 0 of route, status and text are evidently zero vectors, so they don't matter. The unit and rate-unit search was done before this fix, so I'll redo it now with kq 1 and action 1, first on single-START bundles. Then I'll check all drug bundles under both END-attribute settings. Exact zeros would complete the encoder.

```python
res5 = []
for ui in range(6):
    for un in range(6):
        for rr_ in [(2,3),(3,2),(0,0),(1,1)]:
            cfg = {**cfgD, 'unit': {**cfgD['unit'], 'insulin': ui, 'norepinephrine': un}, 'runit': {'insulin': rr_[0], 'norepinephrine': rr_[1]}}
            u_ = u_only_x(cfg, {'kq': 1, 'action': 1})
            zp = mlp_(torch.cat([ctxF, encC['e'], u_], 1), 'z_post', 'gelu')[:, :32]
            er_ = (zp - zm3).abs().mean(1).numpy()
            res5.append((round(er_[so].mean(), 6), ui, un, rr_))
res5.sort(key=lambda r: r[0]); print(res5[:5])
```