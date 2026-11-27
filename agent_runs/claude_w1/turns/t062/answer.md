This result matters for the plan.
- **Original model:** under insulin x2 its effect builds slowly, from −0.5 at hour 13 to −2.3 at hour 24. So the model does carry its own drawn glucose forward, with roughly 0.78 persistence per hour.
- **Truth:** it builds from −3.2 to −18.8.

Under constant extra dose, the true impulse response averages about 9.8 mg/dL per extra u/h over hours 13–24. With 0.78 persistence, a one-hour-lag tilt of about 3 per u/h produces almost the same pooled average. So a lag-1 readout edit could in principle reach the right main readout.

The question is whether the lag-1 dose innovation can be read from what the value head actually sees: the summed view, its normalised query, or the head's hidden layer. If cross-validated R² is around 0.3 or better, the required gain should stay within the factual budget. If it is much lower, I'll need an upstream edit.

```python
Pv=None
act='gelu'
gpc=M2(np.log(dt)[:,None]/10.0,'gap',act); ctx_=outp+hm+gpc
zo_=M2(ctx_,'z_prior',act); view_=ctx_+L(zo_[:,:32],'z_head')
q_=LNf(view_+TE[1],'query_norm')
f8=np.zeros((len(fa),8)); f8[:,0]=fa.has; f8[:,1]=np.where(fa.has>0,np.log1p(fa.lastage.fillna(0)/60)/10,0); f8[:,2]=fa.zl; f8[:,3]=fa.has
base_=CE[fa.code.values]+L(f8,'last_numeric')+VB[fa.lb.values]
wc_=q_+base_
hid_=ACTS[act](L(wc_,'attr_heads.value_bin.0'))
mG=(fa.event=='glucose_finger').values
sub_=fa[mG].copy()
sub_['nu1']=[Ptu2.nu.get((s_,t-1),0) for s_,t in zip(sub_.stay_id,sub_.hour)]
sub_['D1']=[Ptu2.D.get((s_,t-1),0) for s_,t in zip(sub_.stay_id,sub_.hour)]
grp_=sub_.stay_id.values
def cvr(Xm,y):
    Xm=(Xm-Xm.mean(0))/(Xm.std(0)+1e-6); pr=np.zeros_like(y)
    for a,b in GroupKFold(5).split(Xm,y,grp_):
        m=RidgeCV(alphas=np.logspace(-1,4,12)).fit(Xm[a],y[a]); pr[b]=m.predict(Xm[b])
    return round(1-np.mean((y-pr)**2)/np.var(y),3), pr
for nm,F in [('out+meanh',np.hstack([outp[mG],hm[mG]])),('view',view_[mG]),('q',q_[mG]),('hidden',hid_[mG])]:
    r1,_=cvr(F,sub_.nu1.values.astype(float)); r2,_=cvr(F,sub_.D1.values.astype(float))
    print(nm,'R2 nu1',r1,'R2 D1',r2)
```