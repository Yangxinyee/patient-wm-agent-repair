Setting glucose_finger's unit to index 0 (no unit) and glucose_lab's to 5 cuts the measurement-only error to 0.0014, an unusual but data-driven result. It isn't exactly zero yet. Next I'll recompute the error by bundle composition under this setting, and also try glucose_finger at other indices with lab fixed at 5. If the remaining error sits in glucose_finger bundles, something else about the finger encoding is off; if it is spread thinly, it may be numerical.

```python
cfgC = {**cfgB, 'unit': {**cfgB['unit'], 'glucose_finger': 0, 'glucose_lab': 5}}
encC = encode_bundles(A3, cfgC, 'gelu', empty='enc')
zpC = mlp_(torch.cat([ctxF, encC['e'], encC['u']], 1), 'z_post', 'gelu')[:, :32]
errC = (zpC - zm3).abs().mean(1).numpy()
dfE = pd.DataFrame({'comp': comp, 'err': errC, 'ok': prev_ok.numpy()}); dfE = dfE[dfE.ok]
print(dfE.groupby('comp').err.agg(['mean','max','size']).sort_values('size', ascending=False).head(14).round(4).to_string())
```