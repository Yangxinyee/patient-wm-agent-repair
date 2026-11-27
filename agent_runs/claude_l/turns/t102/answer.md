Adding a free output offset doesn't help either. So the stored z is consistent with the z_post layout, and the encodings must be what varies wrongly from bundle to bundle. The residuals are smallest for single arterial-MAP bundles and largest for bundles with glucose_finger or cuff readings.

Next I fit the context offset using **only** the single-atom arterial-MAP bundles at k=1, for each activation choice.
- If one activation fits these to about 1e-5, then the encoder and z_post are right for that atom type.
- Applying that offset to the other bundle types then shows which atom types (glucose_finger, cuff, lab) or which multi-atom combinations are encoded wrongly.

```python
single7=[i for i,s in enumerate(k1) if [a['code'] for a in BUN[s][1][1]]==[7]]
print('n single map_arterial', len(single7))
def fit_sub(ea,za,sub):
    ACTMAP.clear(); ACTMAP.update({'encode.mlp1':ea,'encode.mlp2':ea})
    Ev=np.stack([set_enc(W0,[a for a in BUN[s][1][1] if not a['act']],'encode').numpy() for s in k1])
    Vv=Ev@W1[:,192:384].T+b1
    fun=lambda a: ((act_np(za,a[None,:]+Vv[sub])@W2.T+b2)-Zi[sub]).ravel()
    jac=lambda a: np.concatenate([W2*dact_np(za,a+Vv[i])[None,:] for i in sub],0)
    r=least_squares(fun,np.zeros(192),jac=jac,method='trf',max_nfev=300)
    allres=np.abs((act_np(za,r.x[None,:]+Vv)@W2.T+b2)-Zi).max(1)
    return r, allres
for ea,za in [('gelu','silu'),('silu','silu'),('gelu','gelu'),('silu','gelu'),('relu','silu')]:
    r,allres=fit_sub(ea,za,single7)
    print(ea,za,'fit rms %.6f'%np.sqrt(np.mean(r.fun**2)), 'others median %.4f'%np.median(np.delete(allres,single7)))
```