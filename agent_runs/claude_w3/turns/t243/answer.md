The decay and input-gate parameters (`flow.a`, `a_in`, `delta_in`) are per-dimension vectors of length 192, shared by all four slots. `b_in` and `c_out` are 192×192 linear maps with no bias. So a single dimension index could act as a leaky memory channel in every slot.

Next I'll rank the dimensions by how much h varies across bundles in the saved states (fwd_4), together with their decay rates. I'm looking for a dimension that carries little information and could be repurposed. A dimension with a small spread and a slow decay would be a good candidate. If every dimension has a similar spread, that idea is weakened.

```python
import numpy as np, glob
fs=glob.glob('/work/fwd_4/*.npz'); D=np.load(fs[0]); print(fs[0].split('/')[-1], {k:D[k].shape for k in D.files})
X=[D[k] for k in D.files if D[k].ndim==2 and D[k].shape[1]==992][0]
H=X[:,192:960].reshape(-1,4,192)
sd=H.std(0); mu=H.mean(0)
fa=load('core.flow.a').numpy(); ai=load('core.a_in').numpy(); di=load('core.delta_in').numpy()
sp=np.log1p(np.exp(di)); abar=np.exp(-sp*np.exp(ai)); tau_h=1/np.exp(fa)/3600
sdm=sd.mean(0); o=np.argsort(sdm)
print('sd quantiles',np.round(np.quantile(sdm,[0,.05,.5,.95,1]),2))
for j in o[:10]: print(j, np.round(sd[:,j],2), np.round(mu[:,j],1), 'tau_h %.2g abar %.3f sp %.3f'%(tau_h[j],abar[j],sp[j]))
```