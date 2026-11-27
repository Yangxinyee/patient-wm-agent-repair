gen_samples does not reproduce Xf: there is a constant offset of about 9, and its dose perturbation hardly changes the head inputs. So the old DEC decoders came from another representation, and I cannot rely on them. Xf, EDR and my head replica agree with each other (exact nll0), so I will work in that space. The decisive question is how small the factual noise of the best linear dose readout can be, given the edit responses in EDR. I will compute a plain minimum-variance decoder for each drug: per-row responses from EDR, noise from the within-cell covariance of factual head inputs, over a range of regularisation. It is feasible if its noise sd per unit dose comes out below about 0.6 (insulin) and 0.2 (norepinephrine). If it stays far above that, no head-only correction can meet the factual budget.

```python
import numpy as np
res={}
for drug,cs,key in [('ins',[0,1],('insulin',2.0)),('ne',[2,3],('norepinephrine',2.0))]:
    v=EDR[key]; gi=v['gi']; r=(np.asarray(v['reached'])>0)&np.isin(cf[gi],cs)
    dX=v['dX'][r]; dd=v['dd'][r][:,:3].mean(1)
    dbar=dX.mean(0)/dd.mean()
    msk=np.isin(cf,cs); R=Rf[msk]; S=np.cov(R.T)
    ev=np.linalg.eigvalsh(S)
    for eps in [1e-3,1e-2,1e-1,1.0]:
        Si=np.linalg.inv(S+eps*ev.max()*1e-2*np.eye(192))
        w=Si@dbar; w/=dbar@w
        noise=np.sqrt(w@S@w); resp=(dX@w)/np.maximum(dd,1e-6)
        print(drug,'eps',eps,'noise sd %.3f'%noise,'resp per-row median %.2f IQR %.2f..%.2f'%tuple(np.percentile(resp[dd>0.2],[50,25,75])))
        res[(drug,eps)]=w
print('eig S top/min',np.round(ev[-1],1),np.round(ev[0],5))
```