Run 195 shows that neither b_in nor c_out has a bias, so there is no constant input to hijack for the test.

I don't need one. What I'm testing is how a change in the state y after the action bundle propagates. So I'll add δ (size 10, the same in all four slots) directly to the base state after bundle k0. Then I'll run the next four bundles with the true inputs and compare against the unperturbed run.

The test is the same as before:
- fast-decaying or low-variance y directions should fade within the hour (the next two bundles) if a non-persistent channel is possible;
- random directions show the generic behaviour.

```python
def run_from(Bs,rws,EUs,k0,k1,h0):
    h=h0.clone(); outl=[]
    for k in range(k0,k1+1):
        zk=torch.tensor(SX[rws[k],960:992],dtype=torch.float64)
        dt=(Bs[k][0]-Bs[k-1][0])*3600.0; lo=max(0,k-127)
        Mem_=torch.stack([EUs[j][0] for j in range(lo,k+1)]); Mage_=torch.tensor([(Bs[k][0]-Bs[j][0])*3600.0 for j in range(lo,k+1)],dtype=torch.float64)
        res=core_step_v(h,dt,EUs[k][0],EUs[k][1],zk,Mem_,Mage_); h,o=res[0],res[1]; outl.append((h.clone(),o.clone()))
    return outl
base=run_from(B_,rows_,EU_,k0,k0+4,h0)
ONW=Wt['core.out_norm.weight']; ONB=Wt['core.out_norm.bias']
cands={}
for i in range(2): v_=rng_.standard_normal(192); cands['rand%d'%i]=v_/np.linalg.norm(v_)
for i in [0,5,20]: cands['lowy%d'%i]=lowy[:,i].copy()
v_=np.zeros(192); v_[fast]=1; cands['fast_y']=v_/np.linalg.norm(v_)
v_=lowy@(lowy.T@np.isin(np.arange(192),np.argsort(tau)[:20]).astype(float)); cands['lowy_fastproj']=v_/np.linalg.norm(v_)
for nm_,a_ in cands.items():
    hp=base[0][0]+torch.tensor(a_*10.0)
    o0=Fnn.layer_norm(hp.mean(0),(192,),ONW,ONB,1e-5); do0=float((o0-base[0][1]).norm())
    pert=run_from(B_,rows_,EU_,k0+1,k0+4,hp)
    dy=[10.0]+[float((pert[j][0]-base[j+1][0]).mean(0).norm()) for j in range(4)]
    do=[do0]+[float((pert[j][1]-base[j+1][1]).norm()) for j in range(4)]
    print(nm_,'|dy_mean|',np.round(dy,3),' |dout|',np.round(do,4))
```