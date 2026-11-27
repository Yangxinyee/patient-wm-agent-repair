The earlier run stopped before `build` was defined, so I will re-run the full definition with the decoders taken from DEC2.

```python
import numpy as np, torch
from scipy.special import erf
gel=lambda z:0.5*z*(1+erf(z/np.sqrt(2)))
nW1,nb1,nW2,nTR='attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','embed_transition.value_bin.weight'
vIu=DEC2[('ins','mvdr',-2.8)]; vNu=DEC2[('ne','mvdr',3)]
def build(cI,cN,vI_,vN_,jG=0,jM=114,kG=1.0,kM=4.0,save=False):
    W1e,b1e,W2e,TRe=W1.copy(),b1.copy(),W2.copy(),TR.copy()
    aI=Xf@vI_; aN=Xf@vN_; mI=aI[glu].mean(); mN=aN[mp].mean()
    zI=kG*(aI-mI); zN=kM*(aN-mN)
    BG=np.abs(zI).max()+8; BM=np.abs(zN).max()+8
    W1e[jG]=BG*gw[:192]+kG*vI_; b1e[jG]=BG*gw[192]-kG*mI
    W1e[jM]=-BM*gw[:192]+kM*vN_; b1e[jM]=-BM*gw[192]-kM*mN
    W2e[:,jG]=cI/np.median(var0[cf==0])*tG/kG; W2e[:,jM]=cN/np.median(var0[cf==2])*tM/kM
    P0=Xf@W1.T+b1; P=Xf@W1e.T+b1e
    D=gel(P[:,[jG,jM]])@W2e[:,[jG,jM]].T-gel(P0[:,[jG,jM]])@W2[:,[jG,jM]].T
    for cc in np.unique(cell):
        m=cell==cc; TRe[cc]-=D[m].mean(0)
    L=gel(P)@W2e.T+b2+TRe[cell]
    d=-(logsm(L)[np.arange(N),obf]-lp0)
    print('BG %.1f BM %.1f | min pre glu %.1f max pre map %.1f | dNLL %.4f'%(BG,BM,P[glu,jG].min(),P[mp,jG].max(),d.mean()))
    if save:
        for n,a in zip((nW1,nb1,nW2,nTR),(W1e,b1e,W2e,TRe)):
            save_edit(n,torch.tensor(a,dtype=torch.float32))
        print('edited',edited())
    return L
Le=build(-2.8,3.0,vIu,vNu,save=True)
```