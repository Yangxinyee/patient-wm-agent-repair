The z-sampling noise can be handled. Including it brings the minimum-noise decoders to within-cell noise of about 2.0 (insulin) and 0.37 (NE) per unit response, with z noise 0.14 / 0.035.

At the gains the rollout needs (about 3 per unit, given the model's own feedback through its sampled readings), the factual cost would still be several times the budget. A full-strength version therefore cannot fit within 0.005 nats.

The cost can be cut by applying each correction only at rows where that drug ran in the preceding hour. Elsewhere the edit's dose change is zero, so the correction there is pure cost. I will check how cleanly "drug ran at lag 1" can be detected from the head query, including z noise. A separation of 6 or more standard deviations would allow a clean gate.

```python
offs_=np.r_[0,np.cumsum([len(g['code']) if not isinstance(g['code'],int) else g['code'] for g in GEN])] if False else offs
dI1=np.zeros(len(Xf)); dN1=np.zeros(len(Xf))
for i,g in enumerate(GEN):
    a,bq=offs[i],offs[i+1]; hh_=hf[a:bq].astype(int)
    dI1[a:bq]=g['ins'][np.clip(hh_,0,26)]; dN1[a:bq]=g['ne'][np.clip(hh_,0,26)]
for nm,d1_,rw,codes in [('ins',dI1,ins_rows,[0,1]),('ne',dN1,map_rows,[2,3])]:
    on_=d1_[rw]>0; X_=Xf[rw].astype(np.float64)
    mu1,mu0=X_[on_].mean(0),X_[~on_].mean(0)
    Sw_=np.cov(np.vstack([X_[on_]-mu1,X_[~on_]-mu0]).T)+np.cov(DZ[np.isin(CZ,codes)].T)+1e-3*np.eye(192)
    g_=np.linalg.solve(Sw_,mu1-mu0); p1,p0=X_[on_]@g_,X_[~on_]@g_
    print(nm,'on frac',round(on_.mean(),3),"d'",round((p1.mean()-p0.mean())/np.sqrt(g_@Sw_@g_),2),'overlap: on<max0',round((p1<np.percentile(p0,99.5)).mean(),4))
```