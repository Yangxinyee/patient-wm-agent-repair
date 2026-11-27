The z-dimension route is too weak:
- Δq at readings averages 0.049. Per unit z shift the response is about 0.03.
- The best signal-to-noise per unit shift is below 1 at every lag.
- A linear model of propagation explains only 29% of Δq.

To reach −2 mg/dL per unit/h without adding factual noise, the shift in a quiet dim would need to be very large, beyond where the network behaves linearly. Quiet z dims are quiet because the network barely uses them.

A better route uses a channel the network does use: inject the dose shock along the **z direction that the model itself uses for "glucose came in lower/higher than predicted"** (and the MAP equivalent for norepinephrine). The model's own dynamics then carry the effect forward through its sampled readings. The writer should store the shock (dose minus expected dose given the state) rather than the raw dose, to keep factual predictions unbiased.

That needs a baseline π̂ predicted from the context at the drug bundle. First test: how well can a linear function of ctx predict the insulin and norepinephrine rates at drug bundles? I'll use the 400 stays already fetched as original-model states.

Insulin should reach an R² close to what the glucose-band policy allows. Norepinephrine depends on the previous rate and may be harder. If ctx doesn't carry the previous norepinephrine rate, the norepinephrine baseline needs another source.

```python
XSo=[];META=[]
for f_ in ['fwd_2','fwd_5','fwd_6','fwd_7']:
    dd_=np.load(f'/work/{f_}/states_original.npz'); XSo.append(dd_['x']); META.append(np.c_[dd_['stay_id'],dd_['hour'],dd_['has_insulin'],dd_['has_norepinephrine'],dd_['n_atoms']])
XSo=np.concatenate(XSo); META=np.concatenate(META); sidA=META[:,0].astype(np.int64); hrA=META[:,1]; hiA=META[:,2]>0; hnA=META[:,3]>0
print('rows',len(XSo),'stays',len(np.unique(sidA)))
kd=np.nonzero((hiA|hnA))[0]; kd=kd[(kd>0)]; kd=kd[sidA[kd-1]==sidA[kd]]
dtd=(hrA[kd]-hrA[kd-1])*3600.0
CTX=np.asarray(view_from(XSo[kd-1], dtd)[1])
I_d=np.array([sidx[int(s)] for s in sidA[kd]]); T_d=np.floor(hrA[kd]).astype(int)
yri=ins_all[I_d,T_d]; yrn=ne_all[I_d,T_d]; yrn_prev=np.array([ne_all[i,t-1] if t>=1 else 0 for i,t in zip(I_d,T_d)])
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import Ridge
for nm_,y_ in [('ins',yri),('ne',yrn),('ne_prev',yrn_prev)]:
    for al_ in [1.0,10.0]:
        pr__=cross_val_predict(Ridge(alpha=al_),CTX,y_,groups=sidA[kd],cv=GroupKFold(5))
        print(nm_,'alpha',al_,'CV R2',np.round(1-np.mean((y_-pr__)**2)/np.var(y_),3))
```