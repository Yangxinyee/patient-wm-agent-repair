Restricting to hours with a MAP reading at t largely removes the specification sensitivity. Minimal and full controls now give −0.20 ± 0.15 and −0.34 ± 0.13, against −1.05 versus −0.40 before. So the no-reading rows were the main contaminant: they are titrated on the previous hour's MAP, as the table confirms.

The cleaned norepinephrine estimate is still negative:

| Subset | Implied ×2 readout (se) |
|---|---|
| Both readings at t | −0.20 (0.24) |
| Arterial only | −0.41 (0.17) |
| Cuff only | −0.32 (0.27) |
| Hours ≥ 12 | −0.18 (0.19) |

Insulin is stable across control sets (+0.90 vs +0.99, se ≈ 0.5–0.6), as expected for a clean instrument.

Both drugs now show the same odd shape: nothing at lags 1–2, then a "worse outcome" association from lag 3 on. Insulin goes with higher glucose and norepinephrine with lower MAP.

A shared delayed pattern for two unrelated drugs makes me suspect a shared measurement artefact rather than two paradoxical drugs. The obvious candidate is the reading schedule. If a larger dose changes whether a reading is taken a few hours later, the set of observed future readings is selected by the dose. That alone creates an association with no drug effect. For insulin, a reading always follows any dose, but larger doses may keep glucose readings going longer or alter when they happen.

Test: regress the presence of a finger reading (insulin rows) and of a MAP reading (norepinephrine rows) at t+k on the innovation at t, with the same cross-fitted controls.
- **Supports the artefact:** presence clearly depends on the innovation.
- **Supports real effects:** presence is independent of it.

```python
t0=time.time()
def presence_lp(trt0,rowmask,Xfeats,pres,K=8,label=''):
    si_,ti_=np.nonzero(rowmask); X_=np.stack([Xfeats[n][si_,ti_] for n in Xfeats],1); fold_=((stay_ids[si_]//7)%2)
    def xf(y):
        pred=np.full(len(y),np.nan); ok=~np.isnan(y)
        for f in (0,1):
            tr=ok&(fold_!=f); te=ok&(fold_==f); pred[te]=HGB(max_iter=200,min_samples_leaf=100).fit(X_[tr],y[tr]).predict(X_[te])
        return pred
    d=trt0[si_,ti_]; ep=d-xf(d); out=[]
    for k in range(1,K+1):
        tt=ti_+k; ok=tt<=25; y=np.full(len(si_),np.nan); y[ok]=pres[si_[ok],tt[ok]].astype(float); yr=y-xf(y); m=~np.isnan(yr)
        b=(ep[m]*yr[m]).sum()/(ep[m]**2).sum(); u=ep[m]*(yr[m]-b*ep[m]); cl=pd.Series(u).groupby(si_[m]).sum()
        se=np.sqrt((cl**2).sum())/(ep[m]**2).sum(); out.append('k%d %.4f(%.4f)'%(k,b,se))
    print(label,' '.join(out))
presF=~np.isnan(GF); presA=~np.isnan(GA); presAny=~(np.isnan(GA)&np.isnan(GC)); presL=~np.isnan(GL)
presence_lp(INS0, irows_pos, ifeats, presF, label='INS -> finger present  ')
presence_lp(INS0, irows_pos, ifeats, presL, label='INS -> lab present     ')
presence_lp(NE0, contR, feats, presA, label='NE -> arterial present ')
presence_lp(NE0, contR, feats, presAny, label='NE -> any MAP present  ')
print('baseline presence rates: finger %.3f lab %.3f art %.3f anyMAP %.3f'%(presF.mean(),presL.mean(),presA.mean(),presAny.mean()))
print('time',round(time.time()-t0,1))
```