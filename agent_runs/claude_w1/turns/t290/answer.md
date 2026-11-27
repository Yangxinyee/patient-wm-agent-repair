Units 0 and 114 of the value head contribute almost nothing and can be reused. A linear gate separates glucose rows from MAP rows almost perfectly (scores ≥ 0.99 vs ≤ −0.98). The Ridge decoders give the dose at lags 1–3. Plan: one reused unit per drug computes the lag-averaged decoded dose, gated to its own reading type. It tilts the bin logits along the bin centres, and embed_transition subtracts its mean in each (code, last bin) cell, so factual predictions stay centred on what the last reading implies. This run checks the replica head against the stored factual means and measures the gate and decoder spread within cells. It also converts gains into logit tilts and estimates the factual NLL cost on the 1,500 training stays. If the within-cell gate spread is small and the NLL cost at gains of about ±3 is under 0.005, I go ahead.

```python
import numpy as np, pandas as pd
from scipy.special import erf
gel=lambda z:0.5*z*(1+erf(z/np.sqrt(2)))
vid=np.array([6,10,7,5])[cf.astype(int)]; lb=lbf.astype(int); cell=vid*17+lb
vc=VOCAB['value_bin_centres']
def getv(tab,name,i):
    if isinstance(tab,dict):
        for k in (name,str(i),i):
            if k in tab: return np.array(tab[k],float)
    return np.array(tab[i],float)
nm={6:'glucose_finger',10:'glucose_lab',7:'map_arterial',5:'map_cuff'}
try:
    C={i:getv(vc,n,i) for i,n in nm.items()}; print({nm[i]:np.round(C[i][[0,7,15]],1).tolist() for i in C})
except Exception as e: print('centres err',type(vc).__name__,repr(e)[:150])
glu=np.isin(cf,[0,1]); mp=~glu
cm=np.stack([C[v] for v in vid])
def probs(L):
    p=np.exp(L-L.max(1,keepdims=True)); return p/p.sum(1,keepdims=True)
def mstat(p):
    pp=p[:,1:]; w=pp.sum(1); mu=(pp*cm).sum(1)/w; var=(pp*(cm-mu[:,None])**2).sum(1)/w; return mu,var
L0r=gel(Xf@W1.T+b1)@W2.T+b2+TR[cell]; p0=probs(L0r); mu0,var0=mstat(p0)
for nmv in ['mt','obf','nll0','base_nll']:
    v=globals().get(nmv)
    if isinstance(v,np.ndarray): print(nmv,v.dtype,np.round([np.nanmin(v),np.nanmax(v)],2), ('corr mu0 %.3f'%np.corrcoef(mu0,v)[0,1] if nmv=='mt' else ''))
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