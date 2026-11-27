Key finding: Hmean carries much better rate signal than `out` (ins R² 0.80 vs 0.71, ne R² 0.51 vs 0.36), and all 4 h slots are identical in R², so Hmean is the natural summary. The **delta analysis** at action bundles is the smoking gun: ΔHmean@ins-action has R²=0.66 for insulin rate — this is a **causal** direction (it is literally the change in Hmean produced by processing an insulin atom). The same for ne (R²=0.43). Reading these directions should move under dose intervention because the intervention changes the u vector that feeds into them.

Plan for run 291: fit the ΔHmean regression, use the resulting w as the value-head reader. The reader is naturally in Hmean-space, so when applied to q (= out + LN(mean_h) + …) it picks up rate through the LN(mean_h) term. Centre the reader bias at the QG mean so the baseline tilt at typical gf readings is near zero. Pick a conservative target of −4 mg/dL and +6.9 mmHg per unit rate_read.

Outcome that confirms: reasonable local slope with gf NLL diff below ≈0.01 nats. Save, then dose_edit.

```python
import numpy as np, torch, pickle
from sklearn.linear_model import Ridge
from scipy.special import erf
npz=np.load('/work/fwd_2/states_original.npz')
hi=npz['has_insulin']; hn=npz['has_norepinephrine']
d=pickle.load(open('/work/rate_regs.pkl','rb')); RINS=d['RINS']; RNE=d['RNE']
order=np.lexsort((SHR,SST)); Xs=SX[order]; sid_s=SST[order]; hi_s=hi[order]; hn_s=hn[order]; rins_s=RINS[order]; rne_s=RNE[order]
same=(sid_s[:-1]==sid_s[1:]); D=Xs[1:]-Xs[:-1]; D=D[same]
HMD=(D[:,192:384]+D[:,384:576]+D[:,576:768]+D[:,768:960])/4
rins_c=rins_s[1:][same]; rne_c=rne_s[1:][same]; hi_c=hi_s[1:][same]; hn_c=hn_s[1:][same]
r_di=Ridge(alpha=100).fit(HMD[hi_c],rins_c[hi_c]); r_dn=Ridge(alpha=100).fit(HMD[hn_c],rne_c[hn_c])
print('ΔHmean R²: ins %.3f ne %.3f, |w| ins %.2f ne %.2f'%(r_di.score(HMD[hi_c],rins_c[hi_c]),r_dn.score(HMD[hn_c],rne_c[hn_c]),np.linalg.norm(r_di.coef_),np.linalg.norm(r_dn.coef_)))
R_i=r_di.coef_; R_n=r_dn.coef_
b_i=-(QG@R_i).mean(); b_n=-(QM@R_n).mean()
r_i_val=QG@R_i+b_i; r_n_val=QM@R_n+b_n
print('rate_read QG ins mean %.2f sd %.2f, QM ne mean %.2f sd %.2f'%(r_i_val.mean(),r_i_val.std(),r_n_val.mean(),r_n_val.std()))
print('cross QM ins %.2f ±%.2f; QG ne %.2f ±%.2f'%((QM@R_i+b_i).mean(),(QM@R_i+b_i).std(),(QG@R_n+b_n).mean(),(QG@R_n+b_n).std()))

def T(x): return np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x, dtype=np.float64)
N0w,N0b,N2w='attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight'
W0=T(load_original(N0w)); b0=T(load_original(N0b)); W2=T(load_original(N2w)); b2=T(load_original('attr_heads.value_bin.2.bias')); ETo=T(load_original('embed_transition.value_bin.weight'))
POS={'gf':[QG,QV['glucose_finger']],'map':[QM,QV['map_arterial']]}
NEG={'gf':[QM]+[QV[c] for c in QV if c!='glucose_finger'],'map':[QG]+[QV[c] for c in QV if c!='map_arterial']}
def sp(dd,w): p=np.concatenate([X@w for X in POS[dd]]); n=np.concatenate([X@w for X in NEG[dd]]); return (p.min()+n.max())/2,(p.min()-n.max())/2
mid_g,half_g=sp('gf',WG); mid_m,half_m=sp('map',GD['map'])
cg=np.array(VC['glucose_finger'],float); cm=np.array(VC['map_arterial'],float); tg=np.r_[0.0,cg-cg.mean()]; tm=np.r_[0.0,cm-cm.mean()]
Gd=12.0; s=0.2; U4=[171,114,181,74]
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def head(X,A0_,a0_,A2_): return gelu(X@A0_.T+a0_)@A2_.T+b2
def sm(L): L=L-L.max(1,keepdims=True); P=np.exp(L); return P/P.sum(1,keepdims=True)
def mvar(L,c): P=sm(L); m=P[:,1:]@c; return m,P[:,1:]@(c*c)-m*m
lv=GR['lv'].astype(float).values; lb=np.where(np.isnan(lv),0,1+np.digitize(np.nan_to_num(lv),(cg[1:]+cg[:-1])/2)); ETg=ETo[6*17+lb]
m0g,vg=mvar(head(QG,W0,b0,W2)+ETg,cg); m0m,vm=mvar(head(QM,W0,b0,W2),cm)
tgt={'gf':-4.0,'map':6.9}; lam={'gf':tgt['gf']/vg.mean(),'map':tgt['map']/vm.mean()}; print('lam',lam)
A0=W0.copy(); a0=b0.copy(); A2=W2.copy()
for dd,uA,uB,ws,md,hd,Rd,bd,td,lm in (('gf',U4[0],U4[1],WG,mid_g,half_g,R_i,-b_i,tg,lam['gf']),('map',U4[2],U4[3],GD['map'],mid_m,half_m,R_n,-b_n,tm,lam['map'])):
    A0[uA]=Gd*ws/hd+s*Rd; a0[uA]=-Gd*md/hd-s*bd; A0[uB]=Gd*ws/hd; a0[uB]=-Gd*md/hd
    A2[:,uA]=(lm/s)*td; A2[:,uB]=-(lm/s)*td
dm_g=mvar(head(QG,A0,a0,A2)+ETg,cg)[0]-m0g; dm_m=mvar(head(QM,A0,a0,A2),cm)[0]-m0m
print('gf dm mean %.2f sd %.2f slope %.2f'%(dm_g.mean(),dm_g.std(),np.polyfit(r_i_val,dm_g,1)[0]))
print('map dm mean %.2f sd %.2f slope %.2f'%(dm_m.mean(),dm_m.std(),np.polyfit(r_n_val,dm_m,1)[0]))
be=(cg[1:]+cg[:-1])/2; bidx=np.where(np.isnan(lv),0,1+np.digitize(np.nan_to_num(lv),be))
def nll(X,A0_,a0_,A2_,bi,ET):
    L=head(X,A0_,a0_,A2_)+ET; Lm=L.max(1,keepdims=True); LL=Lm.squeeze()+np.log(np.exp(L-Lm).sum(1))
    return (-(L[np.arange(len(bi)),bi]-LL)).mean()
dnll=nll(QG,A0,a0,A2,bidx,ETg)-nll(QG,W0,b0,W2,bidx,ETg); print('gf NLL diff %.4f'%dnll)
if abs(dnll)<0.02:
    def ts(nm,arr):
        o_=load_original(nm); save_edit(nm,torch.tensor(np.asarray(arr),dtype=o_.dtype) if hasattr(o_,'detach') else np.asarray(arr,dtype=np.float32))
    ts(N0w,A0); ts(N0b,a0); ts(N2w,A2); print('edited:',edited())
else: print('NOT SAVED')
open('/work/notes.md','a').write(f"- Run 291: use ΔHmean@action-bundle regression (ins R² .66, ne .43) as value-head reader. lam {lam}, gf NLL diff {dnll:.4f}.\n")
```