z_head is a linear map from 32 dimensions into the head input. In the tool z is sampled four times, but in my replica (Xf) it was fixed, so Xf has almost no variance along those 32 directions. The minimum-variance decoders and the gate therefore probably loaded on exactly those directions, and the real model's z noise was amplified. The last_numeric inputs (8 columns) were also only approximated in my replica. The fix is to make the gate and decoders exactly orthogonal to these subspaces, which makes them immune to z draws and to register mismatch. This run measures the current sensitivity to z, rebuilds the gate and decoders in the complement, and reports margins, decoder noise and the replica cost. If the old vectors show large z loadings and the new ones keep a clean gate with similar noise, this explains the failure.

```python
import numpy as np
Zw=np.asarray(load_original('z_head.weight'),float); LNw=np.asarray(load_original('last_numeric.weight'),float)
gapn=[(n,SHAPES[n]) for n in NAMES if 'gap' in n or 'type' in n]; print(gapn[:6])
B=[Zw,LNw]+[np.asarray(load_original(n),float).reshape(192,-1) for n,sh in gapn if n.endswith('weight') and sh[0]==192 and len(sh)==2 and sh[1]<=4]
Q,_=np.linalg.qr(np.concatenate(B,1)); Pp=np.eye(192)-Q@Q.T; print('excluded dims',Q.shape[1])
for nmv,vv in [('gate',gw[:192]),('vI',vIu),('vN',vNu)]:
    print(nmv,'z-load %.2f lastnum-load %.2f norm %.1f'%(np.linalg.norm(Zw.T@vv),np.linalg.norm(LNw.T@vv),np.linalg.norm(vv)))
XP=Xf@Pp
y=np.where(glu,1.0,-1.0); A=np.c_[XP,np.ones(N)]; gw2=np.linalg.lstsq(A,y,rcond=None)[0]; s2=A@gw2
gw2=np.r_[Pp@gw2[:192],gw2[192]]
print('gate2 glu min %.3f map max %.3f | z-load %.2e'%(s2[glu].min(),s2[mp].max(),np.linalg.norm(Zw.T@gw2[:192])))
RP=Rf@Pp
DEC3={}
for drug,cs,key in [('ins',[0,1],('insulin',2.0)),('ne',[2,3],('norepinephrine',2.0))]:
    v=EDR[key]; gi=v['gi']; r=(np.asarray(v['reached'])>0)&np.isin(cf[gi],cs)
    dbar=Pp@(v['dX'][r].mean(0)/v['dd'][r][:,:3].mean())
    S=np.cov(RP[np.isin(cf,cs)].T)
    for rg in [3.6e-5,3.6e-4,3.6e-3]:
        w=np.linalg.lstsq(S+rg*np.eye(192),dbar,rcond=None)[0]; w=Pp@w; w/=dbar@w
        print(drug,'rg',rg,'noise %.3f'%np.sqrt(w@S@w),'z-load %.1e'%np.linalg.norm(Zw.T@w)); DEC3[(drug,rg)]=w
```