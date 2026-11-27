The stored search results show that in restart 3 the best match (mean absolute error 1.45, close to the noise floor) came from a column layout using a feature `zl2` in columns 1, 4 and 7. I don't remember what `zl2` is. The same feature in three columns looks like a stand-in for something real rather than the true layout, so I need to identify it.

Next I relate `zl2` to the candidate quantities: the register z, the current observed value's z, the previous reading of the partner code, and the time since the last reading. It will most likely track one of them. If it tracks the current observation's z, the head sees information from the same bundle. If it tracks the register z under another normalisation, the z-score definition is what to fix.

```python
z2=np.asarray(feats2['zl2'],float); zl_=np.asarray(feats2['zl'],float); hs_=np.asarray(feats2['has'],float)
zobs,_=zb([CNAME[c] for c in fa.code.values],fa.observed.values)
print('zl2 stats', np.round([z2.mean(),z2.std(),z2.min(),z2.max()],3), 'frac nonzero', (z2!=0).mean())
for nm_,v_ in [('zl',zl_),('has',hs_),('zobs',zobs),('la_h',np.asarray(feats2['la_h'],float)),('age_h',fa.age_s.values/3600)]:
    m_=hs_>0
    print(nm_, 'corr(all)', round(np.corrcoef(z2,v_)[0,1],3), 'corr(has)', round(np.corrcoef(z2[m_],v_[m_])[0,1],3))
print(pd.DataFrame({'zl':zl_[:8],'zl2':z2[:8],'has':hs_[:8],'last':fa['last'].values[:8],'ev':fa.event.values[:8]}).to_string())
```