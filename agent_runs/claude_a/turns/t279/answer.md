That settles the target. Insulin bundles fall into groups by the rate of the starting atom (kind 0), which is constant within each group. My target had been the rate of the infusion that was ending (kind 2), and the encoder doesn't carry that rate at all: bundles with only an ending atom all share one encoding. So the dose enters the model only through the start atom's rate, and the memory should add that rate when an infusion starts. I'll check how long infusions run (end hour minus start hour), since if they mostly last 1 hour, adding the start rate once corresponds to the hourly running rate the truth kernel uses. Then I'll refit the decoder on the start-atom rate (0 elsewhere), report its residuals, and rebuild. I expect residuals of about 0.05 unit/h or less.

```python
ti=tr[tr['event']=='insulin']; dur=(ti['end_hour']-ti['hour']).round(2)
print('insulin dur', dur.value_counts().head(5).to_dict(), 'n', len(ti))
tn=tr[tr['event']=='norepinephrine']; durn=(tn['end_hour']-tn['hour']).round(2); print('NE dur', durn.value_counts().head(5).to_dict())
yk=r0.reindex(rows_b).fillna(0).values.astype(float)
rg=Ridge(alpha=1e-4).fit(u[:,keepN],yk); pk=rg.predict(u[:,keepN])
hasS=yk>0
for lab_,m_ in [('start solo',hasS&(brN==0)),('start+ne',hasS&(brN>0)),('end-only',(brI>0)&~hasS),('ne-only',(brN>0)&(brI==0)),('nodrug',~isd)]:
    print(lab_,'n=%d resid sd %.3f mean %.3f'%(m_.sum(), np.std((yk-pk)[m_]), np.mean((yk-pk)[m_])))
dvecN=rg.coef_.astype(np.float32); d0N=float(rg.intercept_)
```