Run 230 found the second overwritten global. `atom_vec` reads its embedding tables from a global `W`, which once held a tensor cache keyed by name and is now a 192×32 array from a later analysis. `mk_best` itself wants the event name.

I'll restore `W` as a lazy tensor cache keyed by name and rerun the check:
- the z posterior rebuild error at reading bundles;
- every unit's maximum pre-activation over reading and drug bundles.

A small error at reading bundles would confirm the observation encoder. Units whose maximum stays well below −4 everywhere are the ones I can repurpose for the writer.

```python
class _WD(dict):
    def __missing__(self,k):
        v=load(k).numpy().astype(np.float64); self[k]=v; return v
W=_WD()
try:
    Er=np.stack([e_read(b_) for b_ in rbk]) if ok_form else None
except Exception as ex: Er=None; print('e_read fail',str(ex)[:120])
ok_form='name'
try:
    Er=np.stack([np.asarray(setenc_my([mk_best(ev,v) for ev,v in RB[b_]],'encode'),dtype=np.float64).ravel() for b_ in rbk])
    dtb=np.maximum((S2['hour'][rbk]-S2['hour'][rbk-1])*3600.0,60.0); Cb=ctx_vec(rbk-1,dtb)
    inpR=np.column_stack([Cb,Er,np.tile(np.asarray(U_EMPTY,dtype=np.float64),(len(rbk),1))]); preR=inpR@Wz0.T+bz0
    zzR=gelu_np(preR)@Wz2[:32].T+bz2[:32]; print('reading bundles n',len(rbk),'z err mean',np.round(np.linalg.norm(zzR-X2[rbk,960:992],axis=1).mean(),4))
    EU=[enc_e_var((int(stA[dr[sub_[j]]]),round(float(hrA[dr[sub_[j]]]),3)),'none',True) for j in range(len(sub_))]
    inpD=np.column_stack([CTA[sub_],np.stack([a for a,_ in EU]),np.stack([b for _,b in EU])]); preD2=inpD@Wz0.T+bz0
    mxall=np.maximum(preR.max(0),preD2.max(0)); od=np.argsort(mxall)
    print('lowest max pre (all):',od[:8],np.round(mxall[od[:8]],2)); print('reading-only lowest',np.argsort(preR.max(0))[:6])
except Exception as ex: print('fail',str(ex)[:200])
```