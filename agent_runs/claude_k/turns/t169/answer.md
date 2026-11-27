There is already a `U_EMPTY` array in the kernel, my earlier rebuild of the empty action encoding. The listing also shows the decoder training data (UA/YA) and `setenc_my`.

Now I'll check whether `U_EMPTY` explains the offsets. The fwd_8 numbers imply p_ins·u_empty ≤ −9 (insulin unit fully off). They imply p_ne·u_empty ≈ −0.5, since gelu(5 + x) = 4.49.

For each drug bundle in fwd_8 I'll also invert gelu to get the true projections p·u. I'll compare them with my rebuilt `setenc_my(drug_atoms(...))`. A close match on both projections means the rebuild is right apart from the empty-set convention. A loose match means the rebuild itself is off on drug bundles.

```python
from numpy import interp
gg=np.linspace(-0.75,40,20001); gv=gelu_np(gg)
def ginv_(y): return interp(y,gv,gg,left=np.nan)
print('p_ins.U_EMPTY',float(p_ins_@U_EMPTY),'p_ne.U_EMPTY',float(p_ne_@U_EMPTY),'|U_EMPTY|',np.linalg.norm(U_EMPTY))
imp_ins=ginv_((dz[:,18]+1.5)/0.3)-5; imp_ne=ginv_((dz[:,28]+10)/2)-5
iI=np.nonzero(hi|hn)[0]
rec_i=[];rec_n=[];ok_=[]
for j_ in iI:
    try:
        V_=drug_atoms(sidx[int(sid8[j_])], int(np.floor(hr8[j_])), 'both')
        u_=np.asarray(setenc_my(V_,'encode_action')) if len(V_)>0 else np.zeros(192)
        rec_i.append(p_ins_@u_); rec_n.append(p_ne_@u_); ok_.append(j_)
    except Exception as ex:
        if len(ok_)==0 and len(rec_i)==0: print('err',repr(ex)[:300]); break
rec_i=np.array(rec_i); rec_n=np.array(rec_n); ok_=np.array(ok_)
if len(ok_):
    for nm_,imp_,rec_,m_ in [('ins',imp_ins,rec_i,hi),('ne',imp_ne,rec_n,hn)]:
        mm_=m_[ok_]&np.isfinite(imp_[ok_])
        a_=imp_[ok_][mm_]; b_=rec_[mm_]
        print(nm_,'n',mm_.sum(),'corr',np.round(np.corrcoef(a_,b_)[0,1],3),'mean imp',np.round(a_.mean(),3),'mean rec',np.round(b_.mean(),3),'rmse',np.round(np.sqrt(((a_-b_)**2).mean()),3))
        rr_=(ri if nm_=='ins' else rn)[ok_][mm_]
        for lo_,hi_ in [(0,0.01),(0.4,1.1),(1.4,2.1),(2.4,3.1),(3.4,10)] if nm_=='ins' else [(0,0.001),(0.001,0.2),(0.2,0.5),(0.5,0.9),(0.9,1.01)]:
            k_=(rr_>=lo_)&(rr_<hi_)
            if k_.sum(): print('  rate',lo_,hi_,'n',k_.sum(),'imp',np.round(a_[k_].mean(),2),'rec',np.round(b_[k_].mean(),2),'true rate',np.round(rr_[k_].mean(),2))
```