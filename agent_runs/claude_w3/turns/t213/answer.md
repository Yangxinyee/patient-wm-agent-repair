The core has per-dimension decays (flow.a, a_in, delta_in, each 192), a dense 192×192 `c_out` and `b_in`, two recurrent block layers with self- and cross-attention, and one coda layer. Whether a dimension can hold its own memory depends on `c_out` being close to the identity and on the per-dimension decay rates. I will compute both now. A diagonal-dominant `c_out`, plus some dimensions with slow decay, would support the dedicated-memory route.

```python
import numpy as np
Co=load('core.c_out.weight').numpy(); fa=load('core.flow.a').numpy(); ai=load('core.a_in').numpy(); di=load('core.delta_in').numpy()
sp=np.log1p(np.exp(di)); abar_rate=sp*np.exp(ai)
print('c_out diag mean/sd', Co.diagonal().mean().round(3), Co.diagonal().std().round(3), 'offdiag rms', np.sqrt((Co**2).sum()/192**2 - (Co.diagonal()**2).sum()/192**2).round(3), 'fro dist to I', np.linalg.norm(Co-np.eye(192)).round(2))
tau_flow=1/np.exp(fa)/3600  # hours (dt in s)
print('flow tau h quantiles', np.quantile(tau_flow,[0,.1,.25,.5,.75,.9,1]).round(3))
print('abar=exp(-r): r quantiles', np.quantile(abar_rate,[0,.1,.5,.9,1]).round(3), 'softplus(delta) q', np.quantile(sp,[0,.5,1]).round(3))
ev=np.linalg.eigvals(Co); print('c_out |eig| q', np.quantile(np.abs(ev),[0,.5,.9,1]).round(3))
```