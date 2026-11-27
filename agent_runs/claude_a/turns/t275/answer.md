The drug key is `'insulin'` (an event name in the atoms table), not `'ins'`. Dose bundles can be identified from the atoms that belong to each bundle row. I'll fit the rate decoder on all bundles, with the insulin rate as the target and zero where there is no insulin. That forces the constant "empty" encoding of drug-free bundles to decode to zero, so the memory isn't written at every bundle. Then I'll build on dims 79/99/50 with a 2.5 h half-life and check the factual cost on both sets. I expect a decoder R² of about 0.99, most insulin bundles decoding within ±0.1 unit/h, and costs ≤0.001.

```python
brI=np.asarray(bundle_rates(dsE,'insulin')); brN=np.asarray(bundle_rates(dsE,'norepinephrine'))
rows_drug=set(at.loc[at['event'].isin(['insulin','norepinephrine']),'row'].values)
isd=bd['row'].isin(rows_drug).values
print('ins>0 frac %.3f, ne>0 frac %.3f, drug-bundle frac %.3f'%((brI>0).mean(),(brN>0).mean(),isd.mean()))
X=u[:,keepN]; y=brI
rg=Ridge(alpha=1e-3).fit(X,y); pr=rg.predict(X)
for lab_,m_ in [('ins',brI>0),('ne-only',(brN>0)&(brI==0)),('nodrug',~isd)]:
    print(lab_, 'n=%d resid sd %.3f mean %.3f'%(m_.sum(), np.std((y-pr)[m_]), np.mean((y-pr)[m_])))
dvecN=rg.coef_.astype(np.float32); d0N=float(rg.intercept_)
PA=build_ins2(PO, dvecN, d0N, keepN, jU=79, jR=99, jI=50, half_life_h=2.5, cb=0.4)
print('cost sel2 %+.5f  dsE %+.5f'%(dcost(PA,ds_s2,a_s2), dcost(PA,dsE,a_E)))
```