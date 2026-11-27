There are 2371 drug bundles, all covered by my START/END lookups. The prelude reconstruction ran for 2552 reading and 2371 drug bundles. The prelude FFN has no dead units (the lowest maximum pre-activation is −0.06), so the gate and two innovation units will have to replace the least-used existing units. This run gathers the pieces for that:
1. A cost ranking: each unit's activation spread times its output-column norm.
2. A check that `u_rec(rate_ins, rate_ne)` reproduces the true drug-bundle u from `enc_e_var`; I need it to encode edited doses.
3. The context-decoder file and the context helper signatures, used to define "expected dose".

If `u_rec` matches to within about 0.05 and a few units have near-zero cost, the plan is feasible.

```python
act_sd=gelu_np(PREF[typ>=0].reshape(-1,768).astype(float)).std(0); cn=np.linalg.norm(F2w,axis=0); cost=act_sd*cn
o=np.argsort(cost); print('cheapest', o[:8], np.round(cost[o[:8]],4), 'median', round(float(np.median(cost)),4))
kd=[k for k in range(len(hr)) if typ[k]==1]
RI=np.array([rmap.get((int(sid[k]),round(float(hr[k]),3),'insulin'),np.nan) for k in kd]); RN=np.array([rmap.get((int(sid[k]),round(float(hr[k]),3),'norepinephrine'),np.nan) for k in kd])
err=[np.abs(np.asarray(u_rec(float(np.nan_to_num(RI[i])),float(np.nan_to_num(RN[i])))).reshape(-1)[:192]-np.asarray(enc_e_var((int(sid[k]),round(float(hr[k]),3)),'none')[1])).max() for i,k in enumerate(kd[:300])]
print('u_rec vs true u: mean maxerr %.4f, max %.4f; nan ins %d ne %d'%(np.mean(err),np.max(err),np.isnan(RI).sum(),np.isnan(RN).sum()))
dz_=np.load('/work/decoders.npz'); print({k:dz_[k].shape for k in dz_.files})
for f in ['ctx_vec','ctx_from','ctx_X','get']: 
    try: print(f, inspect.signature(globals()[f]))
    except Exception: print(f,'?')
```