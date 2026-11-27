The centres load correctly. The run stopped only because one of the stored arrays is boolean, which broke a print. The head replica means were already computed. I will repeat the remaining diagnostics with that print fixed.

```python
import numpy as np, pandas as pd
for nmv in ['mt','obf','nll0','base_nll']:
    v=globals().get(nmv)
    if isinstance(v,np.ndarray):
        vf=v.astype(float); print(nmv,v.dtype,v.shape,np.round([np.nanmin(vf),np.nanmax(vf)],2), ('corr mu0 %.3f'%np.corrcoef(mu0,vf)[0,1] if v.shape==mu0.shape and nmv=='mt' else ''))
wI=DEC['glucose_finger'].coef_.mean(0); bI=float(np.mean(DEC['glucose_finger'].intercept_))
wN=DEC['map_arterial'].coef_.mean(0); bN=float(np.mean(DEC['map_arterial'].intercept_))
dI=Xf@wI+bI; dN=Xf@wN+bN; s=Xf@gw[:192]+gw[192]
def wsd(a,mask):
    d=pd.DataFrame({'c':cell[mask],'a':a[mask]}); return float(np.sqrt(((d.a-d.groupby('c').a.transform('mean'))**2).mean()))
print('dI glu mean %.2f sd %.2f wcell %.3f | on map %.1f..%.1f'%(dI[glu].mean(),dI[glu].std(),wsd(dI,glu),dI[mp].min(),dI[mp].max()))
print('dN map mean %.3f sd %.3f wcell %.3f | on glu %.1f..%.1f'%(dN[mp].mean(),dN[mp].std(),wsd(dN,mp),dN[glu].min(),dN[glu].max()))
print('gate wcell sd glu %.4f map %.4f'%(wsd(s,glu),wsd(s,mp)))
print('median var finger %.1f art %.1f'%(np.median(var0[cf==0]),np.median(var0[cf==2])))
```