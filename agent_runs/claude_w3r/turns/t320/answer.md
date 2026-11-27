Ridge gives a strong `out`→insulin-rate direction (R²=0.72 at α=100; |w|=1.6) and a moderate one for norepi (R²=0.37). I'll use those as the value-head rate reads. I also need to drop the core edits so the regression stays valid: it was fit on the **original** model's `out`, and my core edits shift the state distribution.

Plan for run 288: drop the 6 core-tensor edits; keep only value-bin edits. Build the value head with (1) the existing gf/map gates, (2) the new ridge rate reads (R=coef, bias=-intercept), (3) a per-unit tilt scaled so the per-reading effect is −4 mg/dL per unit insulin and +6.9 mmHg per unit norepi (double the prior calibration, to overcome compression through rollout feedback). Check cross-code gate leakage is small and save.

Outcome that confirms: dose_edit after this gives insulin-x2 and ne-x2 shifts of clear opposite sign to original and moving toward targets.

```python
import numpy as np, torch, pickle
from sklearn.linear_model import Ridge
from scipy.special import erf
d=pickle.load(open('/work/rate_regs.pkl','rb')); OUT_=d['OUT_']; ok=d['ok']; RINS=d['RINS']; RNE=d['RNE']
for nm in ['core.prelude.0.attn.kv.weight','core.prelude.0.attn.out.weight','core.prelude.0.ffn.0.weight','core.prelude.0.ffn.0.bias','core.prelude.0.ffn.2.weight','core.b_in.weight']: drop_edit(nm)
r_i=Ridge(alpha=100).fit(OUT_[ok],RINS[ok]); r_n=Ridge(alpha=100).fit(OUT_[ok],RNE[ok])
R_i,b_i=r_i.coef_,-r_i.intercept_; R_n,b_n=r_n.coef_,-r_n.intercept_
print('ins int %.3f |w| %.2f R² %.3f'%(r_i.intercept_,np.linalg.norm(R_i),r_i.score(OUT_[ok],RINS[ok])))
print('ne  int %.3f |w| %.2f R² %.3f'%(r_n.intercept_,np.linalg.norm(R_n),r_n.score(OUT_[ok],RNE[ok])))
POS={'gf':[QG,QV['glucose_finger']],'map':[QM,QV['map_arterial']]}
NEG={'gf':[QM]+[QV[c] for c in QV if c!='glucose_finger'],'map':[QG]+[QV[c] for c in QV if c!='map_arterial']}
def sp(dd,w): p=np.concatenate([X@w for X in POS[dd]]); n=np.concatenate([X@w for X in NEG[dd]]); return (p.min()+n.max())/2,(p.min()-n.max())/2
mid_g,half_g=sp('gf',WG); mid_m,half_m=sp('map',GD['map']); print('half gf %.3f map %.3f'%(half_g,half_m))
cg=np.array(VC['glucose_finger'],float); cm=np.array(VC['map_arterial'],float); tg=np.r_[0.0,cg-cg.mean()]; tm=np.r_[0.0,cm-cm.mean()]
def T(x): return np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x, dtype=np.float64)
N0w,N0b,N2w='attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight'
W0=T(load_original(N0w)); b0=T(load_original(N0b)); W2=T(load_original(N2w)); b2=T(load_original('attr_heads.value_bin.2.bias')); ETo=T(load_original('embed_transition.value_bin.weight'))
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def head(X,A0_,a0_,A2_): return gelu(X@A0_.T+a0_)@A2_.T+b2
def sm(L): L=L-L.max(1,keepdims=True); P=np.exp(L); return P/P.sum(1,keepdims=True)
def mvar(L,c): P=sm(L); m=P[:,1:]@c; return m,P[:,1:]@(c*c)-m*m
lv=GR['lv'].astype(float).values; lb=np.where(np.isnan(lv),0,1+np.digitize(np.nan_to_num(lv),(cg[1:]+cg[:-1])/2)); ETg=ETo[6*17+lb]
m0g,vg=mvar(head(QG,W0,b0,W2)+ETg,cg); m0m,vm=mvar(head(QM,W0,b0,W2),cm); print('var gf %.0f map %.1f'%(vg.mean(),vm.mean()))
Gd=12.0; s=0.2; U4=[171,114,181,74]; tgt={'gf':-4.0,'map':6.9}
lam={'gf':tgt['gf']/vg.mean(),'map':tgt['map']/vm.mean()}; print('lam',lam)
A0=W0.copy(); a0=b0.copy(); A2=W2.copy()
for dd,uA,uB,ws,md,hd,Rd,bd,td,lm in (('gf',U4[0],U4[1],WG,mid_g,half_g,R_i,b_i,tg,lam['gf']),('map',U4[2],U4[3],GD['map'],mid_m,half_m,R_n,b_n,tm,lam['map'])):
    A0[uA]=Gd*ws/hd+s*Rd; a0[uA]=-Gd*md/hd-s*bd; A0[uB]=Gd*ws/hd; a0[uB]=-Gd*md/hd; A2[:,uA]=(lm/s)*td; A2[:,uB]=-(lm/s)*td
for dd,uA,uB in (('gf',U4[0],U4[1]),('map',U4[2],U4[3])):
    for i,Y in enumerate(NEG[dd]):
        pa=Y@A0[uA]+a0[uA]; pb=Y@A0[uB]+a0[uB]; gab=gelu(pa).reshape(-1,1)*A2[:,uA]+gelu(pb).reshape(-1,1)*A2[:,uB]; print('%s NEG[%d] max|logit|=%.2e pre_uA_max=%.2f'%(dd,i,np.abs(gab).max(),pa.max()))
r_g=QG@R_i-b_i; r_m=QM@R_n-b_n
dm_g=mvar(head(QG,A0,a0,A2)+ETg,cg)[0]-m0g; dm_m=mvar(head(QM,A0,a0,A2),cm)[0]-m0m
print('gf dm mean %.3f sd %.3f slope %.3f'%(dm_g.mean(),dm_g.std(),np.polyfit(r_g,dm_g,1)[0]))
print('map dm mean %.3f sd %.3f slope %.3f'%(dm_m.mean(),dm_m.std(),np.polyfit(r_m,dm_m,1)[0]))
def ts(nm,arr):
    o_=load_original(nm); save_edit(nm,torch.tensor(np.asarray(arr),dtype=o_.dtype) if hasattr(o_,'detach') else np.asarray(arr,dtype=np.float32))
ts(N0w,A0); ts(N0b,a0); ts(N2w,A2)
print('edited:',edited())
open('/work/notes.md','a').write(f"- Run 288: dropped core edits; rebuilt value head with ridge-based rate reads (ins R² 0.72, ne R² 0.37) at target -4 mg/dL/unit (gf) and +6.9 mmHg/unit (map). Lam={lam}. Local slopes {np.polyfit(r_g,dm_g,1)[0]:.3f} gf, {np.polyfit(r_m,dm_m,1)[0]:.3f} map.\n")
```