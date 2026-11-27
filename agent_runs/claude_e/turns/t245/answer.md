The insulin readout is slightly asymmetric on recorded data: slope 0.56 per unit of positive dev1, 0.44 for negative. That roughly matches x1/2 coming out at about 80% of what x2 implies. I could fix it with an extra unit that adds gain only on the negative side. But that only shifts error between x2 and x1/2 at about the same factual cost, so I will leave insulin as it is.

For NE x2 there is an approach that should cost almost nothing on factual fit. Read the current NE rate (not its deviation) from the head input on map_arterial rows. Then add a unit that fires only when that reading exceeds about 1.1–1.2. Recorded rates never pass 1.0, so the unit should stay silent on factual data. Under x2, rates near 1.0 become 2.0, and the unit would raise MAP if the readout keeps rising past the training range.

First I check the readout itself:
- how well per-draw head inputs predict the current NE rate (cross-validated by stay);
- whether it stays linear up to 1.0 or flattens;
- how widely it scatters at rate 1.0, since that scatter sets the threshold.

A flattening curve or wide scatter would rule this out before I spend a forward pass.

```python
from sklearn.model_selection import GroupKFold
trN=pd.read_csv('/data/train.csv',usecols=['stay_id','hour','event','rate'])
trN=trN[(trN.event=='norepinephrine')&(trN.stay_id.isin(np.unique(A_.stay.values)))]
gN={s:(g.hour.values,g.rate.values) for s,g in trN.groupby('stay_id')}
ne_rate=np.zeros(len(A_))
for i in range(len(A_)):
    if sts[i] in gN:
        hh_,rr_=gN[sts[i]]; k_=(hh_<hrs[i])&(hh_>=hrs[i]-1.0)
        if k_.any(): ne_rate[i]=rr_[k_][np.argmax(hh_[k_])]
ma_=(A_.code.values==7); idx=np.where(ma_)[0]
print('MA rows',len(idx),'with NE running',(ne_rate[idx]>0).mean().round(3),'rate=1.0 frac',(ne_rate[idx]==1.0).mean().round(3))
Xm=WCall[:,idx,:].astype(np.float64); y=ne_rate[idx]; grp=sts[idx]
mu_=Xm.reshape(-1,192).mean(0); Xc_=Xm-mu_
lam=1e2; pred=np.zeros_like(y); 
for trn,tst in GroupKFold(5).split(idx,y,grp):
    A2=Xc_[:,trn,:].reshape(-1,192); yy=np.tile(y[trn],16)
    wv=np.linalg.solve(A2.T@A2+lam*len(yy)/1e4*np.eye(192),A2.T@(yy-yy.mean()))
    pred[tst]=(Xc_[:,tst,:]@wv).mean(0)+yy.mean()
print('CV R2',(1-((y-pred)**2).mean()/y.var()).round(3))
Tn=pd.DataFrame({'y':y,'p':pred}); Tn['b']=pd.cut(Tn.y,[-.01,0,.1,.25,.5,.75,.95,1.0])
print(Tn.groupby('b',observed=True).agg(n=('p','size'),p=('p','mean'),sd=('p','std'),q95=('p',lambda s: s.quantile(.95)),q99=('p',lambda s: s.quantile(.99))).round(3).to_string())
```