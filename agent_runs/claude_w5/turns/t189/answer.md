Injecting the fixed mean direction moves the next prediction steadily:
- **Glucose:** about −0.30 mg/dL per unit of injection, against −0.36 for a real reading change acting through the state.
- **MAP:** about +0.22 per unit, against +0.29.

The injection does not reproduce per-bundle variation (correlation about 0). But the readouts and factual NLL depend mainly on getting the average right, so the fixed direction is usable.

**Scale.** Matching the estimated next-hour effects needs injection amplitudes of about 6.5 per u/h of insulin and 14.5 per mcg/kg/min of norepinephrine, before any rollout calibration. This comes from −1.96 mg/dL per u/h and +3.18 mmHg per mcg/kg/min, divided by the per-unit shifts above.

Next, the centring term (expected dose given what the model knows), fit at the h:05 decision bundles:
- **Insulin:** the mean rate within glucose bands of the most recent finger reading. The policy is a step function of the measured value.
- **Norepinephrine:** a ridge regression of the rate on the pre-drug state, since that state summarises the reading history the model conditions on.

I will check the fits on held-out stays. A high held-out R² for insulin from the bands, and a moderate one for norepinephrine from the state, would confirm the plan.

```python
from sklearn.linear_model import Ridge
dec=[d for d in D if abs((d['t']%1)-1/12)<1e-3 and d['t']>=0]
for d in dec:
    BB=bundles_B[d['sid']]; gl=np.nan
    for kk in range(d['k']-1,-1,-1):
        c=[a for a in BB[kk][1] if a['code']=='glucose_finger' and a['kind']==0]
        if c: gl=c[0]['value']; break
    d['gl']=gl
edges=[-1e9,150,180,220,260,300,1e9]
G=np.array([d['gl'] for d in dec]); RI=np.array([d['ri'] for d in dec]); RN=np.array([d['rn'] for d in dec])
band=np.digitize(np.nan_to_num(G,nan=100),edges[1:-1])
fit=np.array([d['sid'] in FIT for d in dec])
cI=np.array([RI[fit&(band==b)].mean() for b in range(6)]); print('cI by band',np.round(cI,2),'n',[int((band==b).sum()) for b in range(6)])
chI=cI[band]
r2=lambda y,p:1-((y-p)**2).mean()/y.var()
print('insulin R2 band fit/val %.3f %.3f'%(r2(RI[fit],chI[fit]),r2(RI[~fit],chI[~fit])))
Xs=np.stack([torch.cat([RK[d['sid']][d['k']]['out0'],RK[d['sid']][d['k']]['h0'].mean(0)]).numpy() for d in dec])
mu,sd=Xs[fit].mean(0),Xs[fit].std(0)+1e-6; Z=(Xs-mu)/sd
for lam in [1,10,100]:
    mN=Ridge(alpha=lam).fit(Z[fit],RN[fit]); mI=Ridge(alpha=lam).fit(Z[fit],RI[fit])
    print(lam,'NE state R2 val %.3f'%r2(RN[~fit],mN.predict(Z[~fit])),'| ins state R2 val %.3f'%r2(RI[~fit],mI.predict(Z[~fit])))
print('n dec',len(dec),'mean RN %.3f sd %.3f'%(RN.mean(),RN.std()))
```