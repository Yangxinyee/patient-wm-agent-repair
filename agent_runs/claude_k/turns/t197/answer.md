The innovation regression worked partly. Posterior-minus-prior z at reading bundles is explained by reading surprises with pooled R² 0.36. The directions are distinct:
- **Glucose finger:** 0.0225 per mg/dL.
- **MAP arterial:** 0.049 per mmHg.
- Their cosine is −0.21.

These are channels the network actually uses, so injecting the dose shock along them should move predictions much more than the quiet dims did.

Injecting the raw dose would double count the effect the model already infers from readings, so the injection must be dose minus a ctx baseline. That baseline has to be switched off on reading bundles. The clean gated design needs **three** z_post units: one per drug, plus a shared gate-cancelling unit. Units 55 and 109 are free; I need one more that is nearly constant and can be swapped for its mean.

To rank units, I need trustworthy z_post inputs. I'll check my rebuild of the full posterior on reading and drug bundles of the 99 stays, using ctx from `view_from`, my measurement encoder for e, and the now-exact u (U_EMPTY when no drug). If the error is small, I'll rank units by activation variance times their output weight.

```python
co=zmean.__code__; print('zmean args',co.co_varnames[:co.co_argcount],'globals',co.co_names)
co=meas_atoms.__code__; print('meas_atoms args',co.co_varnames[:co.co_argcount],'globals',co.co_names[:30])
kk_all=np.arange(1,len(sid8)); kk_all=kk_all[sid8[kk_all-1]==sid8[kk_all]]
dt_all=(hr8[kk_all]-hr8[kk_all-1])*3600.0
CTXb=np.asarray(view_from(X8o[kk_all-1], dt_all)[1])
isrd=np.abs(hr8[kk_all]-np.round(hr8[kk_all]))<1e-6
errs_=[]
for j__ in range(0,len(kk_all),7):
    k__=kk_all[j__]; s__=int(sid8[k__]); h__=hr8[k__]
    if isrd[j__]:
        try: V_=meas_atoms(s__, int(round(h__)))
        except Exception as ex: print('meas_atoms err',repr(ex)[:120]); break
        e_=np.asarray(setenc_my(V_,'encode')) if len(V_) else E_EMPTY
        u_=U_EMPTY
    else:
        e_=E_EMPTY; u_=u_c(ins_all[sidx[s__],int(np.floor(h__))], ne_all[sidx[s__],int(np.floor(h__))])
    zr_=np.asarray(zmean(CTXb[j__], e_, u_)).ravel()[:32]
    errs_.append((isrd[j__], np.abs(zr_-X8o[k__,960:992]).max()))
errs_=np.array(errs_)
print('z recon max-abs err: reading bundles mean',np.round(errs_[errs_[:,0]==1,1].mean(),4),'drug bundles mean',np.round(errs_[errs_[:,0]==0,1].mean(),4),'n',len(errs_))
```