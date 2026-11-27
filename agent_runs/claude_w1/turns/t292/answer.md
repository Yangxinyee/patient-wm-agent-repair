Two findings. First, mt is a boolean mask, not a set of means; obf (observed bin) and nll0 (row NLL) are the useful stored arrays. Second, on these head inputs the DEC decoders carry a large offset: mean decoded insulin is 6.4 where the real dose is about 1.5. An offset does not matter, because the per-cell centring removes it. Only the slope (response to a dose change) and the within-cell noise matter, and the noise is small (0.43 and 0.26). The gate is clean (within-cell sd 0.0015). This run checks three things: that my head replica reproduces nll0, how strongly the DEC decoders respond to the four replica edits including leaks, and how decoded dose relates to recorded dose. If the own responses are a sizeable fraction of the dose change and the leaks are small, these decoders are usable.

```python
import numpy as np
nllr=-np.log(p0[np.arange(len(obf)),obf])
print('replica nll vs nll0 mean abs diff %.4f, means %.4f %.4f'%(np.abs(nllr-nll0).mean(),nllr.mean(),nll0.mean()))
for k,v in EDR.items():
    gi=v['gi']; dX=v['dX']; r=np.asarray(v['reached'])>0; c=cf[gi]
    for nmv,w,cs in [('dI',wI,[0,1]),('dN',wN,[2,3])]:
        m=np.isin(c,cs)&r
        print(k,nmv,'resp %.3f'%(dX[m]@w).mean(),'dd123 %.3f'%v['dd'][m][:,:3].mean())
for nmv,act,dec,msk in [('ins',globals().get('dI1'),dI,glu),('ne',globals().get('dN1'),dN,mp)]:
    if isinstance(act,np.ndarray):
        a=act[msk].astype(float); d=dec[msk]; ok=np.isfinite(a)
        sl=np.polyfit(a[ok],d[ok],1); print(nmv,'decoded~actual lag1 slope %.3f icpt %.3f r %.3f'%(sl[0],sl[1],np.corrcoef(a[ok],d[ok])[0,1]),'act mean %.3f'%a[ok].mean())
```