No pooling or size variant helps; the residual pattern stays the same. So the aggregation is probably right, and the mismatch lies in how z is formed from the z_post output. One possibility my fit cannot absorb: the stored posterior mean is the z_post output plus something constant across these stays, such as the prior mean. A residual parameterisation (posterior = prior + correction) is common. The prior depends only on ctx, which is identical here, so it would act as a constant output offset outside the activation, where my free input offset cannot reach it.

Next I add a free 32-dimensional output offset to the fit. If the residual collapses to near zero, the stored z includes a context-dependent additive term. I can then check whether that offset equals the z_prior mean at this ctx. If nothing changes, the encodings themselves are still wrong.

```python
def fit_ac(ea,za,pool='sqrt',size='log1p',free_scale=False):
    ACTMAP.clear(); ACTMAP.update({'encode.mlp1':ea,'encode.mlp2':ea})
    Ev=np.stack([set_enc_v(W0,[a for a in BUN[s][1][1] if not a['act']],'encode',pool,size).numpy() for s in k1])
    Vv=Ev@W1[:,192:384].T+b1
    n=len(k1)
    def fun(p):
        a=p[:192]; c=p[192:224]
        return ((act_np(za,a[None,:]+Vv)@W2.T+b2+c[None,:])-Zi).ravel()
    def jac(p):
        a=p[:192]
        Ja=np.concatenate([W2*dact_np(za,a+Vv[i])[None,:] for i in range(n)],0)
        Jc=np.tile(np.eye(32),(n,1))
        return np.concatenate([Ja,Jc],1)
    return least_squares(fun,np.zeros(224),jac=jac,method='trf',max_nfev=300)
for ea,za in [('gelu','silu'),('silu','silu'),('gelu','gelu')]:
    r=fit_ac(ea,za); rr=np.abs(r.fun.reshape(len(k1),32)).max(1)
    print(ea,za,'rms %.5f median-max %.4f max %.4f'%(np.sqrt(np.mean(r.fun**2)),np.median(rr),rr.max()), ' |c| %.3f'%np.linalg.norm(r.x[192:]))
```