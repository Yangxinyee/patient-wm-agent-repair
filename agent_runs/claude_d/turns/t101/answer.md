Runs 88 and 89 disagree. The bundle after an insulin bundle gave a surprise R² of 0.43 in run 88 and −0.02 in run 89, even though both should target the same dose. The only difference is that run 88 took the dose hour from the later bundle's own time, while run 89 took it from the insulin bundle. So for some next bundles the two hours differ. That would mean run 88's 0.43 measured the surprise of a later dose, at a bundle where that dose had just been seen. I'll check the time gaps between the two bundles and how often the two dose hours disagree, and recompute R² for each case separately.

```python
ii=np.where(hi)[0]; ii=ii[(ii+1<len(X))]; ii=ii[sid3[ii+1]==sid3[ii]]; jj=ii+1
ti=np.floor(hr3[ii]-5/60+1e-9).astype(int); tj=np.floor(hr3[jj]-5/60+1e-9).astype(int)
dt_=np.round((hr3[jj]-hr3[ii])*60).astype(int)
print('gap minutes counts', pd.Series(dt_).value_counts().head(6).to_dict())
print('t0 differs frac', np.mean(tj!=ti), ' next-row has_ins frac', z3['has_insulin'][jj].mean(), 'has_ne', z3['has_norepinephrine'][jj].mean())
for nm,msk in [('same_t',tj==ti),('diff_t',tj!=ti)]:
    rows=jj[msk]; y=Y[rows,0]; Xr=X[rows][:,192:960]; pr=np.zeros(len(y))
    for a,b in kf.split(Xr): pr[b]=Ridge(alpha=30).fit(Xr[a],y[a]).predict(Xr[b])
    print(nm,len(rows),'R2',round(1-np.mean((y-pr)**2)/np.var(y),2),'var y',round(np.var(y),3))
```