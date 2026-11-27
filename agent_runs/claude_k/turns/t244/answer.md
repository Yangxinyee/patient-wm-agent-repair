The best END-atom encoding so far, no attributes in the observation set, rebuilds 4-atom bundles to about 0.012–0.018 and 2-atom bundles to 0.03–0.07. That is not exact, but I may not need it to be.

The writer units will read only ctx and the START-atom action encoding u, and u rebuilds exactly (single-atom bundles have error 0.0014). Their weights on e will be zero. I need e only to judge which existing z_post hidden units are dead everywhere, so that repurposing them costs nothing, and for that an approximate e is enough.

Next I'll rebuild e at reading bundles from the readings via `mk_best` and check the z posterior error there. Then I'll compute every unit's maximum pre-activation over reading and drug bundles. I'm looking for at least three units that stay well below about −4 on all bundle types.

```python
from collections import defaultdict
RB=defaultdict(list)
for i in range(len(R)): RB[int(R.k.values[i])+1].append((R.event.values[i],float(R.value.values[i])))
rbk=np.array(sorted(RB.keys())); rbk=rbk[::3]
def e_read(b_): return np.asarray(setenc_my([mk_best(ev,v) for ev,v in RB[b_]],'encode'),dtype=np.float64).ravel()
try:
    Er=np.stack([e_read(b_) for b_ in rbk])
    dtb=np.maximum((S2['hour'][rbk]-S2['hour'][rbk-1])*3600.0,60.0); Cb=ctx_vec(rbk-1,dtb)
    inpR=np.column_stack([Cb,Er,np.tile(np.asarray(U_EMPTY,dtype=np.float64),(len(rbk),1))]); preR=inpR@Wz0.T+bz0
    zzR=gelu_np(preR)@Wz2[:32].T+bz2[:32]; print('reading bundles n',len(rbk),'z err mean',np.round(np.linalg.norm(zzR-X2[rbk,960:992],axis=1).mean(),4))
    EU=[enc_e_var((int(stA[dr[sub_[j]]]),round(float(hrA[dr[sub_[j]]]),3)),'none',True) for j in range(len(sub_))]
    inpD=np.column_stack([CTA[sub_],np.stack([a for a,_ in EU]),np.stack([b for _,b in EU])]); preD2=inpD@Wz0.T+bz0
    mxall=np.maximum(preR.max(0),preD2.max(0)); od=np.argsort(mxall)
    print('lowest max pre (all bundles):',od[:8],np.round(mxall[od[:8]],2)); print('units 55,109 max',np.round(mxall[[55,109]],2))
except Exception as ex: print('fail',str(ex)[:200])
```