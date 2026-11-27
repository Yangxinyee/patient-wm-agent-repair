No type comes close. The error against fwd_4's predicted means is 9–33 mg/dL, which is as large as the error against the actual readings, so a structural piece of my head is wrong. Before searching the many undocumented choices blindly, I'll measure which components dominate. I'll check the prior z spread, since large sampling noise would make an exact match impossible. I'll also check the norms of each view term, and the correlation and spread of my predicted means against the tool's for the best types. A high correlation with a level offset points to a missing additive term, such as the registers. A low correlation points to the state or view assembly.

```python
i=gfm.prev.values; dt=gfm.dt_s.values
ms=(HH[i]*np.exp(-dt[:,None,None]*fa[None,None,:])).mean(1); on=LNf(ms,'core.out_norm')
ctx=OUT[i]+on+gapf(np.log(dt)/10); zp=Lin(gelu(Lin(ctx,'z_prior.0')),'z_prior.2')
print('prior sd pct',np.percentile(np.exp(zp[:,32:]),[5,50,95]).round(2))
print('norms out,on,gap,zhead',[np.linalg.norm(a,axis=1).mean().round(2) for a in [OUT[i],on,gapf(np.log(dt)/10),Lin(zp[:,:32],'z_head')]])
pm=gfm.predicted_mean.values; print('tool pm sd',pm.std().round(1))
for t in [1,6,7,10]:
    m_,_=head(gfm,t); print(t,'corr',np.corrcoef(m_,pm)[0,1].round(3),'sd',m_.std().round(1),'bias',(m_-pm).mean().round(1))
```