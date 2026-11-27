The leak is identified, and it is not in z_post:
- At the first bundle (admission, zero state), z18 and z28 are exactly 0.
- Yet the edited state already differs from the original by 51% after that bundle.
- The edited attention weights respond to the first bundle's slot tokens (just the slot embeddings, since the state is zero) at about 4,000. Normal slot tokens give about 220.
- This perturbation fades over the next 3–4 bundles. That explains why 98% of the damage sits in hours −1 to 1.
- Elsewhere the gating is clean: the channel is exactly 0 at drug-free bundles.

The fix is to remove the first-bundle slot-token directions from the edit's input space. That stops the misfire exactly. I will then rebuild the edit with separate gains for the insulin and norepinephrine channels.

I'll also check for saturation. The layer-norm on the z token caps large deviations, which would explain why ×2 was too weak (−7.2) while ×½ was nearly right (+5.6). If needed, I'll shrink the deviation's input column and raise the gain so the response is closer to linear.

For this candidate:
- insulin gain set from the saturation curves to aim at −12.5 / +6.25;
- norepinephrine gain raised ×8 as a probe, since it currently does nothing.

This is confirmed if the next factual test shows the early damage gone, i.e. total NLL change at most 0.005.

```python
import numpy as np
kvn='core.prelude.0.attn.kv.weight'
W0=np.asarray(load_original(kvn),dtype=np.float64); DWf=np.load('/work/DW_kv_full.npy').astype(np.float64); A=DWf[336:384]
g1=np.asarray(load('core.prelude.0.norm1.weight'),dtype=np.float64); b1=np.asarray(load('core.prelude.0.norm1.bias'),dtype=np.float64)
def LN(v):
    m=v.mean(-1,keepdims=True); va=v.var(-1,keepdims=True); return (v-m)/np.sqrt(va+1e-5)*g1+b1
SL=np.asarray(load('core.slots'),dtype=np.float64); X=LN(SL)
Q,_=np.linalg.qr(X.T); Ap=A-(A@Q)@Q.T
Wz=np.asarray(load('z_in.weight'),dtype=np.float64); bz=np.asarray(load('z_in.bias'),dtype=np.float64)
Z=x1[:,960:992].copy(); Z0=Z.copy(); Z0[:,18]=0; Z0[:,28]=0; base=Z0@Wz.T+bz
rows=np.where(hi|hn)[0]; nrows=np.where(~hi&~hn&(bi>=1))[0]
Lb=LN(base[rows]); w18=Wz[:,18].copy(); w28=Wz[:,28].copy()
def gdir(w): return ((LN(base[rows]+0.05*w)-Lb)/0.05).mean(0)
def curve(M,w,ts):
    d0=(LN(base[rows]+0.05*w)-Lb)@M.T/0.05; r=[]
    for t in ts:
        d=(LN(base[rows]+t*w)-Lb)@M.T; r.append(np.mean((d*d0).sum(1)/(t*(d0*d0).sum(1)+1e-12)))
    return np.array(r), np.linalg.norm(d0,axis=1).mean()
ts18=np.array([-4,-2,-1,1,2,4,6,9.]); ts28=np.array([-1,-.5,.5,1,2,3.])
for M,nm in [(A,'A'),(Ap,'Ap')]:
    r18,n18=curve(M,w18,ts18); r28,n28=curve(M,w28,ts28); print(nm,'|s18|',round(n18,1),r18.round(2),'|s28|',round(n28,1),r28.round(2))
rA,_=curve(A,w18,ts18)
c18=None
for c in [1,.6,.4,.25,.15]:
    rc,_=curve(Ap,c*w18,ts18); print('c18',c,rc.round(2))
    if c18 is None and rc.min()>=0.85: c18=c
c18=c18 or .15
c28=None
for c in [1,.5,.25]:
    rc,_=curve(Ap,c*w28,ts28); print('c28',c,rc.round(2))
    if c28 is None and rc.min()>=0.85: c28=c
c28=c28 or .25
rN,_=curve(Ap,c18*w18,ts18)
rx2=lambda r:(r[4]+r[5])/2; rh=lambda r:(r[1]+r[2])/2
kI=float(np.sqrt((12.5/7.2)*rx2(rA)/rx2(rN)*(6.25/5.65)*rh(rA)/rh(rN))); kN=8.0
sI=A@gdir(w18); sN=A@gdir(w28)
R=np.linalg.svd(Ap,full_matrices=False)[2][:2]
G=np.stack([gdir(c18*w18),gdir(c28*w28)],1); RG=R@G
Anew=np.stack([kI*sI,kN*sN],1)@np.linalg.inv(RG)@R
prev=np.where(bi>=1)[0]-1
injn=np.array([np.linalg.norm(Anew@LN(x0[prev,192+192*j:384+192*j]+SL[j]).T,axis=0) for j in range(4)])
uI=sI/np.linalg.norm(sI); uN=sN/np.linalg.norm(sN)
nz=LN(base[nrows])@Anew.T; nzA=LN(base[nrows])@A.T
print('kI',round(kI,2),'kN',kN,'c18',c18,'c28',c28,'condRG',round(np.linalg.cond(RG),1),'|sI|',round(np.linalg.norm(sI),1),'|sN|',round(np.linalg.norm(sN),1))
print('first slot inj',np.linalg.norm(Anew@X.T,axis=0).round(2),'normal slot med',np.median(injn,1).round(1),'p99',np.percentile(injn,99,1).round(1))
print('ztok noise sd new I,N',(nz@uI).std().round(1),(nz@uN).std().round(1),'old',(nzA@uI).std().round(1),(nzA@uN).std().round(1))
Wz2=Wz.copy(); Wz2[:,18]*=c18; Wz2[:,28]*=c28
save_edit('z_in.weight',Wz2.astype(np.float32))
Wn=W0.copy(); Wn[336:384]+=Anew; save_edit(kvn,Wn.astype(np.float32))
np.savez('/work/C1.npz',Anew=Anew,c18=c18,c28=c28,kI=kI,kN=kN,Wz2=Wz2,sI=sI,sN=sN,R=R,G=G)
with open('/work/notes.md','a') as f:
    f.write(f'- run292: LEAK FOUND = kv edit picks up first-bundle slot tokens LN(core.slots) (inj ~4000 vs ~220 normal); z18/z28=0 there. Gating clean (z=0 at no-drug bundles).\n- run293: C1 = kv delta rebuilt in row space of A(I-P_slots), per-drug gains kI={kI:.2f} (x full sens), kN={kN}, z_in col18*{c18}, col28*{c28} (linearize LN). Saved /work/C1.npz. Test factual+dose_edit next.\n')
```