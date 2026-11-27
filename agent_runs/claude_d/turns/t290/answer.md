`zpr` returns a (32, 64) array for a batch of 4,970 contexts, so it evidently expects a different input shape. Rather than guess its convention, I will compute the z prior directly from the `core.z_prior` weights with my own MLP. I check it against my saved prior means (HI['zmu']) and then proceed with the Δwc summary as planned.

```python
print({n:SHAPES[n] for n in NAMES if n.startswith('core.z_prior')})
zp=mlpP(c_o,'core.z_prior'); print('zp shape', zp.shape, 'err mean-part', np.abs(zp[:,:32]-HI['zmu'][ri]).max().round(5))
zo=zp[:,:32]; ze=mlpP(c_e,'core.z_prior')[:,:32]
wo=wc_from(zo,c_o,HI['reg'][ri]); we=wc_from(ze,c_e,HI['reg'][ri]); dW=we-wo
print('wc repro err vs WC0', np.abs(wo-WC0[ri]).max().round(4))
ev=RD.event.values[ri]; nd=np.linalg.norm(dW,axis=1)
for e in np.unique(ev): print(e, 'n',(ev==e).sum(),'mean|dwc|', nd[ev==e].mean().round(3), 'p90', np.percentile(nd[ev==e],90).round(3), 'wc sd/dim', wo[ev==e].std(0).mean().round(3))
dX=np.abs(P1['x'][idx]-P0['x'][idx]); print('state change (mean abs): out', dX[:,:192].mean().round(3), 'h', dX[:,192:960].mean().round(3), 'z', dX[:,960:].mean().round(3), '| h scale', np.abs(P0['x'][idx][:,192:960]).mean().round(3))
```