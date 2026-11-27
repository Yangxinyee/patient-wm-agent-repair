The factual numbers can't be right as they stand. The readout moves factual glucose predictions by about 17 mg/dL RMS, yet NLL "improves" by 0.54 nats per glucose reading, which is impossible for a well-fitted model. I suspect two separate problems:
1. RT's `observed` column may be a flag rather than the reading value, which would make the observed-bin NLL meaningless.
2. The readout extrapolates badly on factual rows, possibly at hours ≤ 12, which have no counterfactual counterpart.

On counterfactual rows the out-of-fold readout recovers about 65–70% of the needed shift; for example x2/h12 gives −8.8 against −12.9.

Next I check what `observed` holds and break the factual changes down by hour. Then I refit with factual rows up-weighted. If `observed` is binary, I'll take reading values from the reading table builder or the factual CSV instead.

```python
print('observed dtype',RT.observed.dtype, 'uniq', RT.observed.nunique(), 'mean %.2f'%RT.observed.astype(float).mean(), '| last_val mean %.1f'%RT.last_val.mean())
fc=pd.read_csv('/work/fwd_2/factual_original.csv'); print(fc.columns.tolist()[:12], fc.shape)
ix,oof,mn=fit_eval(isG,vG,100)
f=NONE[ix]; d=(mn-M.mean_f.values[ix]); h=M.hour.values[ix]
for nm,g in (('h<=6',f&(h<=6)),('6<h<=12',f&(h>6)&(h<=12)),('h>12',f&(h>12))):
    print(nm,'n',g.sum(),'mean %.2f rms %.2f'%(d[g].mean(),np.sqrt((d[g]**2).mean())))
```