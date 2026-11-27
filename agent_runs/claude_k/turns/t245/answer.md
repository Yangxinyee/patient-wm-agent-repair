Run 228 failed on an indexing error inside `mk_best`. It probably expects the integer event code from `CODE`, not the event name.

I'll probe both calling forms on one reading, keep whichever works, and then run the same check as before:
- the z posterior error at reading bundles;
- every unit's maximum pre-activation over reading and drug bundles.

The goal is unchanged: find at least three z_post hidden units that are never active, so the writer can use them at no cost.

```python
ok_form=None
for form in ['name','code']:
    try:
        a_=mk_best('glucose_finger' if form=='name' else CODE['glucose_finger'],251.0); ok_form=form; print('mk_best works with',form,np.shape(a_)); break
    except Exception as ex: print(form,'fails:',str(ex)[:80])
def e_read(b_): return np.asarray(setenc_my([mk_best(ev if ok_form=='name' else CODE[ev],v) for ev,v in RB[b_]],'encode'),dtype=np.float64).ravel()
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