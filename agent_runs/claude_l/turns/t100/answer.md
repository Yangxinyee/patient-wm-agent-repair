The fit is still not exact, and the pattern is telling. The residual is smallest for single-atom bundles (one arterial MAP reading, about 0.06) and grows with the number of atoms: bundles with 2–3 readings reach 0.9. An error that grows with atom count points to the set encoder's pooling (sum/√n, mean, or plain sum) or to its size term (log(1+n), n, or log n).

Next I add a pooling option and refit the free context offset for each pooling and size variant. A variant whose residual collapses for multi-atom bundles as well identifies the correct DeepSets aggregation.

```python
def set_enc_v(W, atoms, pre, pool='sqrt', size='log1p'):
    n=len(atoms)
    m=mlp(W, atom_matrix(W, atoms), pre+'.mlp1')
    s={'sqrt':m.sum(0)/math.sqrt(n),'mean':m.mean(0),'sum':m.sum(0)}[pool]
    sv={'log1p':math.log(1+n),'n':float(n),'log':math.log(n),'zero':0.0}[size]
    s=s+lin(W, torch.tensor([sv],dtype=T), pre+'.size')
    return lnorm(W, mlp(W, s, pre+'.mlp2'), pre+'.norm')
def fit_a2(ea,za,pool,size):
    ACTMAP.clear(); ACTMAP.update({'encode.mlp1':ea,'encode.mlp2':ea})
    Ev=np.stack([set_enc_v(W0,[a for a in BUN[s][1][1] if not a['act']],'encode',pool,size).numpy() for s in k1])
    Vv=Ev@W1[:,192:384].T+b1
    fun=lambda a: ((act_np(za,a[None,:]+Vv)@W2.T+b2)-Zi).ravel()
    jac=lambda a: np.concatenate([W2*dact_np(za,a+Vv[i])[None,:] for i in range(len(k1))],0)
    return least_squares(fun,np.zeros(192),jac=jac,method='trf',max_nfev=200)
for pool in ['sqrt','mean','sum']:
    for size in ['log1p','n','zero']:
        r=fit_a2('gelu','silu',pool,size); rr=np.abs(r.fun.reshape(len(k1),32)).max(1)
        print(pool,size,'rms %.4f  median-max %.4f  max %.4f'%(np.sqrt(np.mean(r.fun**2)),np.median(rr),rr.max()))
```