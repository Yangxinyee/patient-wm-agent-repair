Norepinephrine is dosed differently from insulin. It is always on when the last MAP arterial is below 65, and use tapers off between 65 and 90. The mean rate falls smoothly with MAP (about 0.75 at MAP 30–45, 0.46 at 60, 0.15 at 70), with substantial spread (SD 0.2–0.3). About 72k of the 230k norepinephrine hours occur in stays without an arterial line, presumably dosed on cuff MAP. The readout uses map_arterial, so the arterial-line stays matter most.

Next I will check whether norepinephrine is titrated, i.e. whether r_t depends on r_{t−1} beyond the current MAP, and whether the residual is i.i.d. A strong dependence on the previous rate would mean the identifying variation is the innovation, not the level.

```python
art = N_[N_.m.notna() & (N_.mage==0)].copy()
art['m2']=(art.m-70)**2/100; art['on']=(art.r>0).astype(float); art['rp_on']=(art.rprev>0).astype(float)
b,s_,nn=ols(art,'r',['m','m2','rprev','rp_on']); print('r ~ m + m2 + rprev + rp_on:'); print(b.round(4).to_string()); print(s_.round(4).to_string())
art['m5']=np.floor(art.m/5)*5
art['res']=art.r-art.groupby('m5').r.transform('mean')
# autocorrelation of residual within stay among on-hours
R = np.full((n,25),np.nan)
for t,g in art.groupby('t'): R[g.stay.values, t]=g.res.values
x=R[:,1:-1].ravel(); y=R[:,2:].ravel(); m=~np.isnan(x)&~np.isnan(y)
print('lag1 corr of residual (rate - E[r|m5]):', np.corrcoef(x[m],y[m])[0,1], m.sum())
# rate distribution at m in 50-55 among on
s=art[(art.m>=50)&(art.m<55)&(art.r>0)].r; print('rate dist m 50-55:', s.describe().round(3).to_dict()); print(np.histogram(s,bins=np.linspace(0,1,11))[0])
```