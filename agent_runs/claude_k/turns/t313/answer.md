The run failed because the stored per-bundle rates contain NaN, presumably at drug bundles where one drug is not running. That poisoned the lag features and the least-squares fit.

The q reconstruction itself ran:
- 2,555 reading bundles were found.
- 3,300 s is the usual gap after a drug bundle.
- Insulin is on at 69% of drug bundles, and reading bundles carry no rates, as expected.

I will redo the analysis with NaN set to 0. This run:
- estimates b, the change in q at a reading per unit rate at lags 1 and 2, from the doubled-dose emulation;
- computes the reader efficiency η against the natural q covariance, restricted to directions orthogonal to the value head's per-code register embeddings so the reader cannot pick up the last-value features;
- merges the original model's factual errors (fwd4, same 99 stays) to estimate, per code group, the factual cost or benefit of a reader with a given counterfactual gain;
- prints VOCAB's structure, which I need next run to build the gated reader units.

What would confirm or refute the plan:
- If norepinephrine's reader has a large effective η and costs little at gains of about 2 per mcg/kg/min, while insulin's η is tiny, I build a norepinephrine reader at full strength and an insulin reader at whatever gain the 0.005 budget allows.
- If both are tiny, readers are not viable.

```python
import numpy as np, pandas as pd, json
rI=np.nan_to_num(np.asarray(rateI,float)); rN=np.nan_to_num(np.asarray(rateN,float))
dmap={}
for i in np.where(~isread)[0]: dmap[(st[i],int(np.floor(hour[i]+1e-6)))]=(rI[i],rN[i])
def lag(i,L,j): return dmap.get((st[i],int(round(hour[i]))-L),(0.0,0.0))[j]
F=np.array([[lag(i,1,0),lag(i,2,0),lag(i,1,1),lag(i,2,1)] for i in k])
print('F means',F.mean(0).round(3),'nan dQ',np.isnan(dQ).sum())
X1=np.c_[F,np.ones(len(k))]; b=np.linalg.lstsq(X1,dQ,rcond=None)[0]; R=dQ-X1@b
Sn=np.cov(Qo.T); Sr=np.cov(R.T)
Ereg=np.vstack([A('embed.bins.value.weight'),A('embed.bins.rate.weight'),A('last_numeric.weight').T,A('last_status.weight')])
U_,s_,Vt=np.linalg.svd(Ereg,full_matrices=False); rr=(s_>1e-6*s_[0]).sum(); Pp=np.eye(192)-Vt[:rr].T@Vt[:rr]
def wdir(bv,S,P,lam=1e-2):
    Sp=P@S@P; Si=np.linalg.pinv(Sp+lam*np.trace(Sp)/192*P); w=P@Si@P@bv; return w/(w@bv)
for i,nm in enumerate(['I1','I2','N1','N2']):
    w=wdir(b[i],Sn,Pp); print(nm,'|b| %.3f eta_res %.2f | perp-reader: sd(w.q)=%.3f -> eta_nat %.2f ; w.b(other lags) %s'%(np.linalg.norm(b[i]),b[i]@np.linalg.pinv(Sr+1e-2*np.trace(Sr)/192*np.eye(192))@b[i],(Qo@w).std(),1/(Qo@w).var(),np.round(b[:4]@w,2)))
wI=wdir(b[0],Sn,Pp); wN=wdir(b[2],Sn,Pp)
print('VOCAB keys',list(VOCAB.keys())); print('codes',VOCAB.get('codes'))
fd=pd.read_csv('/work/fwd_4/factual_original.csv'); cols=fd.columns.tolist(); print(cols,len(fd)); print(fd.head(2).to_string())
try:
    sc=[c for c in cols if 'stay' in c][0]; hc=[c for c in cols if 'hour' in c or c=='time'][0]
    ec=[c for c in cols if c in ('event','code','name','event_name')][0]; vc=[c for c in cols if c in ('value','observed','obs','y')][0]; mc=[c for c in cols if 'mean' in c][0]
    pos={(st[i],int(round(hour[i]))):j for j,i in enumerate(k)}
    fd['_j']=[pos.get((s,int(round(h))),-1) for s,h in zip(fd[sc],fd[hc])]; f2=fd[fd._j>=0].copy()
    f2['e']=f2[vc]-f2[mc]; f2['zI']=(Qo@wI)[f2._j]; f2['zN']=(Qo@wN)[f2._j]
    for ev,g in f2.groupby(ec):
        ve=g.e.var()
        for zz in ['zI','zN']:
            cz=np.cov(g.e,g[zz])[0,1]; vz=g[zz].var()
            cost={kap:(kap**2*vz-2*kap*cz)/(2*ve) for kap in (-2,-1,1,2,3)}
            print(ev,len(g),'sd e %.2f'%np.sqrt(ve),zz,'cov %.3f var %.3f'%(cz,vz),'dNLL@kappa',{a:round(c,4) for a,c in cost.items()})
except Exception as ex: print('merge failed', ex)
RD=dict(b=b,Sn=Sn,Sr=Sr,Pp=Pp,wI=wI,wN=wN,F=F)
```