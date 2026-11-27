Run 284 confirmed what the earlier sessions left in the kernel:
- The core edit dictionary is there: 6 tensors, all shapes match.
- The value-head inputs are there: the gf-only selector, the rate read vectors and the per-unit-rate input changes.
- The value head has 192 hidden units. The transition table has 13 codes × 17 rows. glucose_finger is code 6 and map_arterial is code 7.

The run's last line failed only on printing the variable list.

Next, in one run, I will build the value-head part of the drug-response term and save every edit:
- Two of the 8 unused hidden units respond to the insulin rate, on glucose_finger rows only.
- Two more respond to the norepinephrine rate, on map_arterial rows only.
- They tilt the value logits along the bin centres. The starting gains are −2 mg/dL per unit insulin and +3.45 mmHg per unit norepinephrine per step.

Before saving I check three things locally:
- the selector margins,
- that the shift per unit rate matches the chosen gain,
- that the other codes' logits do not change.

I also print the stay lists for the forward requests. The build is confirmed if both selectors have a positive margin and the change on other codes stays below 1e-3. If not, the run does not save.

```python
import numpy as np, json, glob, torch, pandas as pd
from scipy.special import erf
def T(x): return np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x, dtype=np.float64)
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def sm(L): L=L-L.max(1,keepdims=True); P=np.exp(L); return P/P.sum(1,keepdims=True)
N0w,N0b,N2w='attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight'
W0=T(load_original(N0w)); b0=T(load_original(N0b)); W2=T(load_original(N2w)); b2=T(load_original('attr_heads.value_bin.2.bias')); ETo=T(load_original('embed_transition.value_bin.weight'))
def head(X,A0,a0,A2): return gelu(X@A0.T+a0)@A2.T+b2
POS={'gf':[QG,QV['glucose_finger']],'map':[QM,QV['map_arterial']]}
NEG={'gf':[QM]+[QV[c] for c in QV if c!='glucose_finger'],'map':[QG]+[QV[c] for c in QV if c!='map_arterial']}
ALL=np.vstack([QG,QM]+list(QV.values())); pre=ALL@W0.T+b0
IU=[31,74,103,114,124,171,173,181]; print('IU maxpre',np.round(pre[:,IU].max(0),2))
U4=sorted(IU,key=lambda u:pre[:,u].max())[:4]; print('U4',U4,'orig contrib',np.round([np.abs(gelu(pre[:,u])).max()*np.abs(W2[:,u]).max() for u in U4],5))
SEL={}; ok=True
for d,w in (('gf',WG),('map',GD['map'])):
    p=np.concatenate([X@w for X in POS[d]]); n=np.concatenate([X@w for X in NEG[d]])
    mid=(p.min()+n.max())/2; half=(p.min()-n.max())/2; SEL[d]=(w,mid,half); ok&=bool(half>0)
    print(d,'sel pos[%.2f,%.2f] neg[%.2f,%.2f] half %.3f'%(p.min(),p.max(),n.min(),n.max(),half))
dqi=DQ['ins'].mean(0); dqn=DQ['ne'].mean(0); Ri,r0i,_=RR['ins']; Rn,r0n,_=RR['ne']
RD_={'gf':(Ri,r0i,dqi),'map':(Rn,r0n,dqn)}
for d,X in (('gf',QG),('map',QM)):
    R,r0,_=RD_[d]; rd=X@R-r0; ng=np.concatenate([Y@R-r0 for Y in NEG[d]])
    print(d,'read mean %.3f sd %.3f gain ins %.3f ne %.3f negrange [%.1f,%.1f]'%(rd.mean(),rd.std(),dqi@R,dqn@R,ng.min(),ng.max()))
cg=np.array(VC['glucose_finger'],float); cm=np.array(VC['map_arterial'],float)
tg=np.r_[0.0,cg-cg.mean()]; tm=np.r_[0.0,cm-cm.mean()]
lv=GR['lv'].astype(float).values; print('lv q',np.round(np.nanpercentile(lv,[0,5,50,95,100]),2),'nan',int(np.isnan(lv).sum()))
lb=np.where(np.isnan(lv),0,1+np.digitize(np.nan_to_num(lv),(cg[1:]+cg[:-1])/2))
ETg=ETo[6*17+lb] if np.nanmedian(lv)>20 else 0.0
def mvar(L,c): P=sm(L); m=P[:,1:]@c; return m,P[:,1:]@(c*c)-m*m
_,vg=mvar(head(QG,W0,b0,W2)+ETg,cg); _,vm=mvar(head(QM,W0,b0,W2),cm)
print('var gf mean %.0f med %.0f | map mean %.1f med %.1f'%(vg.mean(),np.median(vg),vm.mean(),np.median(vm)))
s=0.2; KT={'gf':-2.0,'map':3.45}
def build(lam):
    A0=W0.copy(); a0=b0.copy(); A2=W2.copy()
    for d,t,uA,uB in (('gf',tg,U4[0],U4[1]),('map',tm,U4[2],U4[3])):
        w,mid,half=SEL[d]; R,r0,_=RD_[d]; ng=np.concatenate([Y@R-r0 for Y in NEG[d]]); Gd=max(12.0,6+s*np.abs(ng).max())
        A0[uA]=Gd*w/half+s*R; a0[uA]=-Gd*mid/half-s*r0; A0[uB]=Gd*w/half; a0[uB]=-Gd*mid/half
        A2[:,uA]=(lam[d]/s)*t; A2[:,uB]=-(lam[d]/s)*t
    return A0,a0,A2
def leak(A0,a0,A2):
    L=0.0
    for d,uA,uB in (('gf',U4[0],U4[1]),('map',U4[2],U4[3])):
        for Y in NEG[d]:
            for r in (0,10):
                h=gelu((Y+r*(dqi+dqn))@A0[[uA,uB]].T+a0[[uA,uB]]); L=max(L,np.abs(h[:,0,None]*A2[:,uA]+h[:,1,None]*A2[:,uB]).max())
    return L
def check(lam):
    A0,a0,A2=build(lam); out={}
    for d,X,c,E in (('gf',QG,cg,ETg),('map',QM,cm,0.0)):
        dq=RD_[d][2]; m1,_=mvar(head(X+dq,A0,a0,A2)+E,c); m0,_=mvar(head(X+dq,W0,b0,W2)+E,c)
        mb1,_=mvar(head(X,A0,a0,A2)+E,c); mb0,_=mvar(head(X,W0,b0,W2)+E,c)
        out[d]=tuple(np.round([(m1-m0).mean(),(m1-m0).std(),(mb1-mb0).mean(),(mb1-mb0).std()],4))
    return out,leak(A0,a0,A2),(A0,a0,A2)
lam={'gf':KT['gf']/vg.mean(),'map':KT['map']/vm.mean()}
o,lk,_=check(lam); lam={d:lam[d]*KT[d]/o[d][0] for d in lam}; o,lk,VH=check(lam)
print('lam',{d:round(v,6) for d,v in lam.items()},'shift/unit (mean,sd) & baseline (mean,sd)',o,'other-code max logit change %.2e'%lk)
def tosave(name,arr):
    o_=load_original(name); save_edit(name, torch.tensor(np.asarray(arr),dtype=o_.dtype) if hasattr(o_,'detach') else np.asarray(arr,dtype=np.float32))
if ok and lk<1e-3:
    A0,a0,A2=VH
    for k,v in EDIT.items(): tosave(k,v)
    tosave(N0w,A0); tosave(N0b,a0); tosave(N2w,A2)
    print('saved',edited()); print('verify %.2e'%max(np.abs(T(load(k))-np.asarray(v,dtype=np.float64)).max() for k,v in list(EDIT.items())+[(N0w,A0),(N0b,a0),(N2w,A2)]))
else: print('NOT SAVED')
LAM=lam; VHB=dict(U4=U4,s=s,SEL=SEL,tg=tg,tm=tm)
for f in ('sel1','selS'):
    x=json.load(open(f'/work/{f}.json')); print(f,type(x).__name__,len(x)); print(json.dumps(x)[:900])
print(glob.glob('/work/fwd_3/*'))
open('/work/notes.md','a').write(f"- Run 285: value-head drug-response units U4={U4} (pairs, s=0.2, code-selective via WG / GD['map']), tilt along centred bin centres; per-step gain -2.0 mg/dL per ins unit, +3.45 mmHg per ne unit; lam={LAM}; saved core EDIT + value_bin.0/.2 if ok={ok}, other-code change {lk:.1e}. No band compensation yet (next: factual both on train -> band means).\n")
```