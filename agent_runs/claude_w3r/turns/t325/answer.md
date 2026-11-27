Local behavior is right-shaped: gf slope −4.3, map slope +7.1 mg/dL per unit rate_read, with tiny mean bias (0.17/0.02). Importantly, NLL on QG actually **improved** by 0.02 nats — the tilt aligns with real rate↔glucose correlations in the training data. My save check excluded exactly `dnll=-0.02`. I'll loosen the check and sweep lam to find the largest scale that keeps gf NLL budget safe, then save.

Outcome that confirms: NLL diff on gf stays below ~+0.004 for a scale up to ~3-5x, letting me push the per-unit tilt without breaking the factual budget.

```python
import numpy as np, torch
from scipy.special import erf
cg=np.array(VC['glucose_finger'],float); cm=np.array(VC['map_arterial'],float); tg=np.r_[0.0,cg-cg.mean()]; tm=np.r_[0.0,cm-cm.mean()]
Gd=12.0; s=0.2; U4=[171,114,181,74]
POS={'gf':[QG,QV['glucose_finger']],'map':[QM,QV['map_arterial']]}
NEG={'gf':[QM]+[QV[c] for c in QV if c!='glucose_finger'],'map':[QG]+[QV[c] for c in QV if c!='map_arterial']}
def sp(d_,w): p=np.concatenate([X@w for X in POS[d_]]); n=np.concatenate([X@w for X in NEG[d_]]); return (p.min()+n.max())/2,(p.min()-n.max())/2
mid_g,half_g=sp('gf',WG); mid_m,half_m=sp('map',GD['map'])
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def head(X,A0_,a0_,A2_): return gelu(X@A0_.T+a0_)@A2_.T+b2
def sm(L): L=L-L.max(1,keepdims=True); P=np.exp(L); return P/P.sum(1,keepdims=True)
def mvar(L,c): P=sm(L); m=P[:,1:]@c; return m,P[:,1:]@(c*c)-m*m
def nll(X,A0_,a0_,A2_,bi,ET):
    L=head(X,A0_,a0_,A2_)+ET; Lm=L.max(1,keepdims=True); LL=Lm.squeeze()+np.log(np.exp(L-Lm).sum(1))
    return (-(L[np.arange(len(bi)),bi]-LL)).mean()
lv=GR['lv'].astype(float).values; be=(cg[1:]+cg[:-1])/2; bidx_g=np.where(np.isnan(lv),0,1+np.digitize(np.nan_to_num(lv),be))
m0g,vg=mvar(head(QG,W0,b0,W2)+ETg,cg); m0m,vm=mvar(head(QM,W0,b0,W2),cm)
nll0_g=nll(QG,W0,b0,W2,bidx_g,ETg); print('nll0_g %.4f var gf %.0f map %.1f'%(nll0_g,vg.mean(),vm.mean()))
best=None; best_rec=None
for scale in [1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.0,10.0]:
    lm_g=(-4.0*scale)/vg.mean(); lm_m=(6.9*scale)/vm.mean()
    A0_=W0.copy(); a0_=b0.copy(); A2_=W2.copy()
    for dd,uA,uB,ws,md,hd,Rd,bd,td,lm in (('gf',U4[0],U4[1],WG,mid_g,half_g,R_i,-b_i,tg,lm_g),('map',U4[2],U4[3],GD['map'],mid_m,half_m,R_n,-b_n,tm,lm_m)):
        A0_[uA]=Gd*ws/hd+s*Rd; a0_[uA]=-Gd*md/hd-s*bd; A0_[uB]=Gd*ws/hd; a0_[uB]=-Gd*md/hd
        A2_[:,uA]=(lm/s)*td; A2_[:,uB]=-(lm/s)*td
    dnll=nll(QG,A0_,a0_,A2_,bidx_g,ETg)-nll0_g
    dm_g=mvar(head(QG,A0_,a0_,A2_)+ETg,cg)[0]-m0g; dm_m=mvar(head(QM,A0_,a0_,A2_),cm)[0]-m0m
    print(f'scale {scale:4.1f}: lam_g {lm_g:+.4f} lam_m {lm_m:+.4f} dnll_g {dnll:+.4f} dm_g {dm_g.mean():+.2f}±{dm_g.std():.2f} dm_m {dm_m.mean():+.2f}±{dm_m.std():.2f}')
    if dnll < 0.003:
        best=scale; best_rec=(A0_.copy(), a0_.copy(), A2_.copy(), lm_g, lm_m)
print('BEST scale:', best)
if best is not None:
    A0,a0,A2,lam_g,lam_m = best_rec
    N0w,N0b,N2w='attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight'
    def ts(nm,arr):
        o_=load_original(nm); save_edit(nm,torch.tensor(np.asarray(arr),dtype=o_.dtype) if hasattr(o_,'detach') else np.asarray(arr,dtype=np.float32))
    ts(N0w,A0); ts(N0b,a0); ts(N2w,A2); print('edited:',edited(),'lam',lam_g,lam_m)
open('/work/notes.md','a').write(f"- Run 292: scale sweep of ΔHmean-based value head. Best scale {best}.\n")
```