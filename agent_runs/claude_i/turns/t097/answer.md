Encoding the empty action set (instead of using zero), with frequency-order units (glucose 3, MAP 2) and kq 0, cuts the measurement-only z error from 0.061 to 0.013. That is still about 5% of the z spread, so one more convention is off. Next I'll break the remaining error down by bundle composition: which measurement codes are present, how many atoms, and whether it is the first bundle after a drug bundle. If the error is concentrated in one code or one atom count, that points to that code's attributes (unit, bins or z-score) or to the size term. If it is spread evenly, it points to ctx (gap or flow).

```python
cfgB = {'unit': {'glucose_finger': 3, 'glucose_lab': 3, 'map_arterial': 2, 'map_cuff': 2, 'insulin': 4, 'norepinephrine': 5}, 'runit': {'insulin': 2, 'norepinephrine': 3}, 'end_attrs': True, 'kq': 0}
encB = encode_bundles(A3, cfgB, 'gelu', empty='enc')
zpB = mlp_(torch.cat([ctxF, encB['e'], encB['u']], 1), 'z_post', 'gelu')[:, :32]
errB = (zpB - zm3).abs().mean(1).numpy()
comp = A3.groupby('bi').code.apply(lambda c: '+'.join(sorted(c))).reindex(range(NB)).fillna('none').values
dfE = pd.DataFrame({'comp': comp, 'err': errB, 'ok': prev_ok.numpy(), 'dt': dt_s.numpy()/60})
dfE = dfE[dfE.ok]
print(dfE.groupby('comp').err.agg(['mean','size']).sort_values('size', ascending=False).head(12).round(4).to_string())
print('by dt (min):', dfE.groupby(pd.cut(dfE.dt, [0, 6, 30, 56, 61, 120, 1e9])).err.mean().round(4).to_dict())
```