There are no `core.z_prior.*` tensors, so the prior lives under another name. I will list the tensor names containing "prior" (and z-related names) with their shapes. Then I try a generic Linear–GELU–Linear evaluation of that block, checked against HI['zmu']. If it matches, the same run computes Δwc. This rebuild of the head inputs is what decides whether the probe's footprint can be read at MAP rows.

```python
zn=[n for n in NAMES if 'prior' in n or 'z_head' in n or 'z_post' in n]; print({n:SHAPES[n] for n in zn})
pn=sorted([n for n in zn if 'prior' in n and n.endswith('weight') and len(SHAPES[n])==2])
def mlp_generic(x,ws):
    for i,wn in enumerate(ws):
        x=x@Wt(wn).T+Wt(wn[:-6]+'bias')
        if i<len(ws)-1: x=gel(x)
    return x
try:
    zp=mlp_generic(c_o,pn); print('layers',pn,'zp',zp.shape,'err',np.abs(zp[:,:32]-HI['zmu'][ri]).max().round(5))
    zo=zp[:,:32]; ze=mlp_generic(c_e,pn)[:,:32]
    wo=wc_from(zo,c_o,HI['reg'][ri]); we=wc_from(ze,c_e,HI['reg'][ri]); dW=we-wo; print('wc repro err vs WC0', np.abs(wo-WC0[ri]).max().round(4))
    ev=RD.event.values[ri]; nd=np.linalg.norm(dW,axis=1)
    for e in np.unique(ev): print(e,'n',(ev==e).sum(),'mean|dwc|',nd[ev==e].mean().round(3),'p90',np.percentile(nd[ev==e],90).round(3),'wc sd/dim',wo[ev==e].std(0).mean().round(3))
except Exception as ex: print('ERR', repr(ex)[:300])
```