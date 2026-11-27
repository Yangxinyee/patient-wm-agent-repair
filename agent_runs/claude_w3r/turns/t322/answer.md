The rollout response is directionally correct now (edited ins_x2 shift −0.358 vs original −0.521; ne_x2 −0.601 vs −1.74) but much smaller than target. Interpretation: my ridge reader, fit on observational data, is confounded with the current glucose/MAP level. During rollout, when my tilt lowers glucose, the predicted-gf feeds back and reduces the reader's output along the gf-confounded direction — so the rate-rate response shrinks. This explains why edited(x2)−edited(record) = −0.36 is less negative than the natural −0.52.

Fix: regress gf-level and MAP-level on `out`, then orthogonalise the rate reader against those directions. The orthogonal direction should still move under rate intervention but not couple to the feedback. Also scale lam modestly up (target −12 mg/dL and +21 mmHg per unit rate). Also measure NLL change on QG.

Outcome that confirms: orthogonal R² for rate is still positive (>0.1), local slope reasonable, QG NLL diff < 0.01, and the next dose_edit shows larger absolute shifts in the correct directions.

```python
import numpy as np, torch, pickle, pandas as pd
from sklearn.linear_model import Ridge
from scipy.special import erf
d=pickle.load(open('/work/rate_regs.pkl','rb')); OUT_=d['OUT_']; ok=d['ok']; RINS=d['RINS']; RNE=d['RNE']
tr=pd.read_csv('/data/train.csv', usecols=['stay_id','hour','event','value'])
S=set(np.unique(SST).tolist()); trs=tr[tr['stay_id'].isin(S)]
def lastval(e):
    evs=trs[(trs['event']==e)&trs['value'].notna()][['stay_id','hour','value']].values
    by={};
    for s,h,v in evs: by.setdefault(int(s),[]).append((h,v))
    for k in by: by[k].sort()
    r=np.full(len(SST),np.nan)
    for i in range(len(SST)):
        s=int(SST[i]); h=SHR[i]; lst=by.get(s,[]); best=np.nan
        for h0,v in lst:
            if h0<=h: best=v
            else: break
        r[i]=best
    return r
GF=lastval('glucose_finger'); MAP=lastval('map_arterial')
msk=ok & ~np.isnan(GF) & ~np.isnan(MAP); print('msk n:', int(msk.sum()))
ri=Ridge(alpha=100).fit(OUT_[msk],RINS[msk]); rn=Ridge(alpha=100).fit(OUT_[msk],RNE[msk])
rg=Ridge(alpha=100).fit(OUT_[msk],GF[msk]); rm=Ridge(alpha=100).fit(OUT_[msk],MAP[msk])
print('R² ins %.3f ne %.3f gf %.3f map %.3f'%(ri.score(OUT_[msk],RINS[msk]),rn.score(OUT_[msk],RNE[msk]),rg.score(OUT_[msk],GF[msk]),rm.score(OUT_[msk],MAP[msk])))
def orth(w,bs):
    w=w.copy()
    for b in bs: w-=(w@b/max(b@b,1e-12))*b
    return w
R_i=orth(ri.coef_,[rg.coef_,rm.coef_]); R_n=orth(rn.coef_,[rg.coef_,rm.coef_])
mu=OUT_[msk].mean(0); b_i=RINS[msk].mean()-R_i@mu; b_n=RNE[msk].mean()-R_n@mu
pri=OUT_[msk]@R_i+b_i; prn=OUT_[msk]@R_n+b_n
r2i=1-((pri-RINS[msk])**2).sum()/((RINS[msk]-RINS[msk].mean())**2).sum()
r2n=1-((prn-RNE[msk])**2).sum()/((RNE[msk]-RNE[msk].mean())**2).sum()
print('orth R²: ins %.3f ne %.3f |w| ins %.2f ne %.2f'%(r2i,r2n,np.linalg.norm(R_i),np.linalg.norm(R_n)))
def T(x): return np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x, dtype=np.float64)
N0w,N0b,N2w='attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight'
W0=T(load_original(N0w)); b0=T(load_original(N0b)); W2=T(load_original(N2w)); b2=T(load_original('attr_heads.value_bin.2.bias'))
ETo=T(load_original('embed_transition.value_bin.weight'))
POS={'gf':[QG,QV['glucose_finger']],'map':[QM,QV['map_arterial']]}
NEG={'gf':[QM]+[QV[c] for c in QV if c!='glucose_finger'],'map':[QG]+[QV[c] for c in QV if c!='map_arterial']}
def sp(d_,w): p=np.concatenate([X@w for X in POS[d_]]); n=np.concatenate([X@w for X in NEG[d_]]); return (p.min()+n.max())/2,(p.min()-n.max())/2
mid_g,half_g=sp('gf',WG); mid_m,half_m=sp('map',GD['map'])
cg=np.array(VC['glucose_finger'],float); cm=np.array(VC['map_arterial'],float); tg=np.r_[0.0,cg-cg.mean()]; tm=np.r_[0.0,cm-cm.mean()]
Gd=12.0; s=0.2; U4=[171,114,181,74]
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def head(X,A0_,a0_,A2_): return gelu(X@A0_.T+a0_)@A2_.T+b2
def sm(L): L=L-L.max(1,keepdims=True); P=np.exp(L); return P/P.sum(1,keepdims=True)
def mvar(L,c): P=sm(L); m=P[:,1:]@c; return m,P[:,1:]@(c*c)-m*m
lv=GR['lv'].astype(float).values; lb=np.where(np.isnan(lv),0,1+np.digitize(np.nan_to_num(lv),(cg[1:]+cg[:-1])/2)); ETg=ETo[6*17+lb]
m0g,vg=mvar(head(QG,W0,b0,W2)+ETg,cg); m0m,vm=mvar(head(QM,W0,b0,W2),cm)
tgt={'gf':-12.0,'map':21.0}; lam={'gf':tgt['gf']/vg.mean(),'map':tgt['map']/vm.mean()}; print('lam',lam)
A0=W0.copy(); a0=b0.copy(); A2=W2.copy()
for dd,uA,uB,ws,md,hd,Rd,bd,td,lm in (('gf',U4[0],U4[1],WG,mid_g,half_g,R_i,-b_i,tg,lam['gf']),('map',U4[2],U4[3],GD['map'],mid_m,half_m,R_n,-b_n,tm,lam['map'])):
    A0[uA]=Gd*ws/hd+s*Rd; a0[uA]=-Gd*md/hd-s*bd; A0[uB]=Gd*ws/hd; a0[uB]=-Gd*md/hd
    A2[:,uA]=(lm/s)*td; A2[:,uB]=-(lm/s)*td
r_g_=QG@R_i+b_i; r_m_=QM@R_n+b_n
dm_g=mvar(head(QG,A0,a0,A2)+ETg,cg)[0]-m0g; dm_m=mvar(head(QM,A0,a0,A2),cm)[0]-m0m
print('gf dm mean %.2f sd %.2f slope %.2f'%(dm_g.mean(),dm_g.std(),np.polyfit(r_g_,dm_g,1)[0]))
print('map dm mean %.2f sd %.2f slope %.2f'%(dm_m.mean(),dm_m.std(),np.polyfit(r_m_,dm_m,1)[0]))
be=(cg[1:]+cg[:-1])/2; bidx=np.where(np.isnan(lv),0,1+np.digitize(np.nan_to_num(lv),be))
def nll(X,A0_,a0_,A2_,bi,ET):
    L=head(X,A0_,a0_,A2_)+ET; Lm=L.max(1,keepdims=True); LL=Lm.squeeze()+np.log(np.exp(L-Lm).sum(1))
    return (-(L[np.arange(len(bi)),bi]-LL)).mean()
print('gf NLL diff %.4f'%(nll(QG,A0,a0,A2,bidx,ETg)-nll(QG,W0,b0,W2,bidx,ETg)))
def ts(nm,arr):
    o_=load_original(nm); save_edit(nm,torch.tensor(np.asarray(arr),dtype=o_.dtype) if hasattr(o_,'detach') else np.asarray(arr,dtype=np.float32))
ts(N0w,A0); ts(N0b,a0); ts(N2w,A2); print('edited:',edited())
open('/work/notes.md','a').write(f"- Run 289: orthogonal rate reader (vs gf/map level). orth R² ins {r2i:.3f}, ne {r2n:.3f}. lam={lam}, gf slope {np.polyfit(r_g_,dm_g,1)[0]:.2f}, map slope {np.polyfit(r_m_,dm_m,1)[0]:.2f}, gf NLL diff {nll(QG,A0,a0,A2,bidx,ETg)-nll(QG,W0,b0,W2,bidx,ETg):.4f}.\n")
```