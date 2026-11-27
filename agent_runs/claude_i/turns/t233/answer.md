Two problems showed up. `core_step` needs ages as a torch tensor; the numpy array I passed caused the crash. My dt also disagrees with the stored `dts` by up to 97,800 s. That is probably only at stay-first bundles, where `run_stay` uses dt = 0 anyway, but I will check by printing the mismatch with first bundles excluded. Then I will rerun the validation. If the mismatch disappears without first bundles and the relative errors are about 1e-6, the pipeline is confirmed.

```python
nf=~first
print('dt mismatch excl first: max',float(np.abs(dtl[nf]-np.asarray(dts,dtype=float)[nf]).max()))
def step_local(k, o_p=None, h_p=None, ztok_add=None):
    if o_p is None: o_p,h_p=prev_state(k)
    dt=float(dtl[k]); hf=h_p*torch.exp(-dt*torch.exp(fa)); ctx=ctx_from(o_p,hf,dt)
    z=zpost_mean(ctx,EA[k],UA[k]); idx=np.arange(max(j0_of[k],k+1-128),k+1)
    ages=torch.tensor((tm[k]-tm[idx])*60.0,dtype=torch.float32)
    if ztok_add is not None:
        b0=WT['z_in.bias'].clone(); WT['z_in.bias']=b0+ztok_add
        try: h,o=core_step(hf,EA[k],UA[k],z,EA[idx],ages,V_)
        finally: WT['z_in.bias']=b0
    else: h,o=core_step(hf,EA[k],UA[k],z,EA[idx],ages,V_)
    return ctx,z,h,o
ez=[];eh=[]
for k in ksv:
    ctx,z,h,o=step_local(int(k)); zt=X3f[k,960:]; ht=X3f[k,192:960]
    ez.append(float((z.reshape(-1)-zt).norm()/zt.norm())); eh.append(float((h.reshape(-1)-ht).norm()/ht.norm()))
print('z relerr med/90',np.round(np.percentile(ez,[50,90]),6),' h relerr med/90/max',np.round(np.percentile(eh,[50,90,100]),6))
```