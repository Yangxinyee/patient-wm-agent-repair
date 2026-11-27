Run 194 failed because `core.b_in` has no bias, so I can't inject into h_mid that way. If `core.c_out` has a bias, adding δ to it at one step injects δ straight into the pre-coda state c = c_out(h_mid). That is just as good for testing persistence: it equals injecting C⁻¹δ into h_mid.

So I'll list the core tensor names and shapes first. Then I'll re-run the persistence test through the c_out bias if it exists. My candidate directions are expressed directly in y space:
- low-variance directions of mean y;
- fast-decaying dimensions;
- random directions.

```python
print([(k,tuple(v.shape)) for k,v in Wt.items() if k.startswith('core.') and ('b_in' in k or 'c_out' in k or 'delta' in k or 'a_in' in k or 'slots' in k or 'z_in' in k)])
if 'core.c_out.bias' in Wt:
    def run_from(Bs,rws,EUs,k0,k1,h0,inject=None,key='core.c_out.bias'):
        h=h0.clone(); outl=[]
        for k in range(k0,k1+1):
            zk=torch.tensor(SX[rws[k],960:992],dtype=torch.float64)
            dt=(Bs[k][0]-Bs[k-1][0])*3600.0; lo=max(0,k-127)
            Mem_=torch.stack([EUs[j][0] for j in range(lo,k+1)]); Mage_=torch.tensor([(Bs[k][0]-Bs[j][0])*3600.0 for j in range(lo,k+1)],dtype=torch.float64)
            bsave=None
            if inject is not None and k in inject:
                bsave=Wt[key].clone(); Wt[key]=Wt[key]+inject[k]
            try: res=core_step_v(h,dt,EUs[k][0],EUs[k][1],zk,Mem_,Mage_)
            finally:
                if bsave is not None: Wt[key]=bsave
            h,o=res[0],res[1]; outl.append((h.clone(),o.clone()))
        return outl
    cands={}
    for i in range(2): v_=rng_.standard_normal(192); cands['rand%d'%i]=v_/np.linalg.norm(v_)
    for i in [0,5,20]: cands['lowy%d'%i]=lowy[:,i].copy()
    v_=np.zeros(192); v_[fast]=1; cands['fast_y']=v_/np.linalg.norm(v_)
    v_=lowy[:,:40]@(lowy[:,:40].T@np.isin(np.arange(192),np.argsort(tau)[:20]).astype(float)); cands['lowy_fastproj']=v_/np.linalg.norm(v_)
    for nm_,a_ in cands.items():
        pert=run_from(B_,rows_,EU_,k0,k0+4,h0,{k0: torch.tensor(a_*10.0)})
        dy=[float((pert[j][0]-base[j][0]).mean(0).norm()) for j in range(5)]
        do=[float((pert[j][1]-base[j][1]).norm()) for j in range(5)]
        print(nm_,'|dy_mean|',np.round(dy,3),' |dout|',np.round(do,4))
```