Bundles holding only MAP readings now match exactly (error 0.0000). So the MAP encoding, ctx, the gap term and the empty-action convention are all right. The remaining errors are in glucose bundles (about 0.009 with finger, 0.03-0.04 with lab) and drug bundles (0.04-0.10). So the glucose unit index is probably wrong, possibly differing between finger and lab readings. Next I'll search the unit index separately for glucose_finger and glucose_lab (0-5) on measurement-only bundles. Exact zeros for one choice would settle the measurement encoder and let me move to the drug encoder.

```python
res2 = []
mo = (prev_ok & (encB['nu'] == 0)).numpy()
for ugf in range(0, 6):
    for ugl in range(0, 6):
        cfg = {**cfgB, 'unit': {**cfgB['unit'], 'glucose_finger': ugf, 'glucose_lab': ugl}}
        enc_ = encode_bundles(A3, cfg, 'gelu', empty='enc')
        zp = mlp_(torch.cat([ctxF, enc_['e'], enc_['u']], 1), 'z_post', 'gelu')[:, :32]
        er_ = (zp - zm3).abs().mean(1).numpy()
        res2.append((round(er_[mo].mean(), 5), ugf, ugl))
res2.sort(); print(res2[:6])
```