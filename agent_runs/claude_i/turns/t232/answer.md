`eu2` works per bundle and returns (e, u), and the weights are float32. Now I will encode all 5156 bundles and run the validation: recomputed z posterior and `core_step` output against the tool's states on 60 random non-first bundles. Median relative errors around 1e-6 would confirm the pipeline (dt and ages in seconds, memory = the stay's last 128 e's including the current one). Errors of order 1 would mean a convention mismatch, such as dt units or memory indexing.

```python
t0=time.time()
N3=len(sid); EA=torch.zeros(N3,192); UA=torch.zeros(N3,192)
for k in range(N3):
    e,u=eu2(atoms_by_bundle[k]); EA[k]=torch.as_tensor(e,dtype=torch.float32); UA[k]=torch.as_tensor(u,dtype=torch.float32)
print('enc time',round(time.time()-t0,1))
X3f=torch.tensor(np.asarray(X3),dtype=torch.float32); fa=WT['core.flow.a']; V_={'incl':1,'unit':'s','age':'both'}
dtl=np.where(first,0.0,np.r_[0.0,np.diff(tm)]*60.0)
print('dt vs dts maxdiff',float(np.abs(dtl-np.asarray(dts,dtype=float)).max()))
def prev_state(k):
    if first[k]: return torch.zeros(192), torch.zeros(4,192)
    return X3f[k-1,:192].clone(), X3f[k-1,192:960].reshape(4,192).clone()
def step_local(k, o_p=None, h_p=None, ztok_add=None):
    if o_p is None: o_p,h_p=prev_state(k)
    dt=float(dtl[k]); hf=h_p*torch.exp(-dt*torch.exp(fa)); ctx=ctx_from(o_p,hf,dt)
    z=zpost_mean(ctx,EA[k],UA[k]); idx=np.arange(max(j0_of[k],k+1-128),k+1); ages=(tm[k]-tm[idx])*60.0
    if ztok_add is not None:
        b0=WT['z_in.bias'].clone(); WT['z_in.bias']=b0+ztok_add
        try: h,o=core_step(hf,EA[k],UA[k],z,EA[idx],ages,V_)
        finally: WT['z_in.bias']=b0
    else: h,o=core_step(hf,EA[k],UA[k],z,EA[idx],ages,V_)
    return ctx,z,h,o
rng=np.random.default_rng(0); ksv=rng.choice(np.where(~first)[0],60,replace=False); ez=[];eh=[]
for k in ksv:
    ctx,z,h,o=step_local(int(k)); zt=X3f[k,960:]; ht=X3f[k,192:960]
    ez.append(float((z.reshape(-1)-zt).norm()/zt.norm())); eh.append(float((h.reshape(-1)-ht).norm()/ht.norm()))
print('z relerr med/90',np.round(np.percentile(ez,[50,90]),6),' h relerr med/90/max',np.round(np.percentile(eh,[50,90,100]),6))
```