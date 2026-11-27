The old decoder gives nonsense on this dose-encoder output (residual sd about 1400), so it was fitted on some other representation; I'll drop it. My new decoder decodes norepinephrine-only and drug-free bundles cleanly as zero. At insulin bundles, though, its error (0.9 unit/h) is about as large as the spread of the rates themselves. A linear readout of a 192-dim encoding of a numeric rate should do far better. Either the encoder carries the rate nonlinearly (rates come in 0.5-unit steps from 0.5 to 5), or there are several insulin atoms per bundle, or something is misaligned. I'll compare:
- the rate spread within insulin bundles;
- the best single-dim correlation with the rate;
- a fit restricted to insulin bundles;
- the number of insulin atoms per bundle.

If one atom per bundle is typical and a fit on insulin bundles alone still explains little, the encoding is nonlinear and I'll decode per rate bin instead.

```python
ys=y[solo]; Xs=u[solo][:,keepN]
c=np.array([np.corrcoef(Xs[:,k],ys)[0,1] if Xs[:,k].std()>1e-6 else 0 for k in range(Xs.shape[1])])
rg2=Ridge(alpha=1e-2).fit(Xs,ys); p2=rg2.predict(Xs)
print('solo ins: rate sd %.3f, max|corr| %.3f, within-solo R2 %.3f resid %.3f'%(ys.std(), np.nanmax(np.abs(c)), 1-np.var(ys-p2)/np.var(ys), np.std(ys-p2)))
ai=at[at['event']=='insulin']; nper=ai.groupby('row').size()
print('insulin atoms per bundle:', nper.value_counts().head(4).to_dict(), '| kinds:', ai['kind'].value_counts().head(4).to_dict())
print('u sd at solo ins (mean over dims) %.3f; distinct u rows among solo: %d'%(Xs.std(0).mean(), len(np.unique(np.round(Xs[:,:5],4),axis=0))))
```