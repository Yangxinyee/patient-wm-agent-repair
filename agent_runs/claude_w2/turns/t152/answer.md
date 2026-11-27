The head is still not exact. My own 4-draw averages differ from the z-expectation by RMSE ~1.75, but the tool differs by 3.23. My NLL is also 0.07 nats worse than the tool's (1.493 vs 1.424). So part of the input definition is still wrong. The weird best assignment, with log-age in five columns, is probably compensating for a wrong register or a wrong constant. To locate the mismatch I'll break the residual (tool − expectation) down by code, by whether a last value exists, by readings per bundle, by age of the last value, and by gap. I'll also check whether the "absent" rows of the bins tables are zero vectors.

```python
cols_b=['la_s','la_s','zl','la_s','zero','la_h','la_h','zero']
PE=full_eval(cols_b,nz=64,seed=1); pmE=(PE[:,1:]*CENr).sum(1).numpy()
nllE=-np.log(PE[np.arange(len(fa4)),obs_bin].numpy())
D=fa4[['event','has','n_in_bundle','age_s','dt','hour','gl_same','partner_same']].copy()
D['r']=tool_pm-pmE; D['dn']=fa4.nll.values-nllE
D['ageb']=pd.cut(D.age_s,[-1,1,3500,3700,7300,1e9]); D['dtb']=pd.cut(D.dt,[0,300,3000,3700,1e9])
for gcol in ['event','has','n_in_bundle','ageb','dtb','gl_same','partner_same']:
    print(D.groupby(gcol,observed=True).agg(n=('r','size'),mr=('r','mean'),rms=('r',lambda x: np.sqrt(np.mean(x**2))),dn=('dn','mean')).round(3))
print('bins row0 norms', [W[f'embed.bins.{a}.weight'][0].norm().item() for a in ['value','rate','dose']])
```