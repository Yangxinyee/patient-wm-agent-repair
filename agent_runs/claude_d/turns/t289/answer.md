The alignment is exact (original vs XALL 0.0002) and ctx reproduction is exact. So I can rebuild the head inputs from edited states. The edited states differ from the original by up to 68 in some component, which is likely in the unnormalised h slots. I will check the relative size after fixing the bug.

The crash came from my assumption about what `zpr` returns. This run inspects its return type, takes the prior mean correctly, and recomputes Δwc by reading type.

```python
zo_=zpr(c_o); print(type(zo_), [np.shape(t) for t in zo_] if isinstance(zo_,(list,tuple)) else np.shape(zo_))
def zmean(z):
    if isinstance(z,(list,tuple)): z=z[0]
    z=np.asarray(z); return z[:, :32] if z.ndim==2 else z
zo=zmean(zo_); ze=zmean(zpr(c_e)); print('zmu err', np.abs(zo-HI['zmu'][ri]).max().round(5))
wo=wc_from(zo,c_o,HI['reg'][ri]); we=wc_from(ze,c_e,HI['reg'][ri]); dW=we-wo
print('wc repro err vs WC0', np.abs(wo-WC0[ri]).max().round(4))
ev=RD.event.values[ri]; nd=np.linalg.norm(dW,axis=1)
for e in np.unique(ev): print(e, 'n',(ev==e).sum(),'mean|dwc|', nd[ev==e].mean().round(3), 'p90', np.percentile(nd[ev==e],90).round(3), 'wc sd/dim', wo[ev==e].std(0).mean().round(3))
dX=np.abs(P1['x'][idx]-P0['x'][idx]); print('state change by block (mean abs): out', dX[:,:192].mean().round(3), 'h', dX[:,192:960].mean().round(3), 'z', dX[:,960:].mean().round(3), '| state scale', np.abs(P0['x'][idx][:,192:960]).mean().round(3))
```