Run 192 failed because `core_step_v` returns more than two values. In run 191 I only indexed `[0]` and `[1]`; `step_at` evidently trims the result to (h, out). I'll take the first two elements and re-run the same test.

As before, the test injects size-10 components along each candidate direction at the insulin action bundle at 11:05. Then it tracks the change in mean y and in out over the next four bundles. A direction whose change falls well below 0.1 of its initial size within an hour is a good non-persistent channel.

```python
def run_from(Bs,rws,EUs,k0,k1,h0,inject=None):
    h=h0.clone(); outl=[]
    for k in range(k0,k1+1):
        zk=torch.tensor(SX[rws[k],960:992],dtype=torch.float64)
        dt=(Bs[k][0]-Bs[k-1][0])*3600.0; lo=max(0,k-127)
        Mem_=torch.stack([EUs[j][0] for j in range(lo,k+1)]); Mage_=torch.tensor([(Bs[k][0]-Bs[j][0])*3600.0 for j in range(lo,k+1)],dtype=torch.float64)
        bsave=None
        if inject is not None and k in inject:
            bsave=Wt['core.b_in.bias'].clone(); Wt['core.b_in.bias']=Wt['core.b_in.bias']+inject[k]/SPD
        try: res=core_step_v(h,dt,EUs[k][0],EUs[k][1],zk,Mem_,Mage_)
        finally:
            if bsave is not None: Wt['core.b_in.bias']=bsave
        h,o=res[0],res[1]; outl.append((h.clone(),o.clone()))
    return outl
base=run_from(B_,rows_,EU_,k0,k0+4,h0)
print('n returned by core_step_v', len(core_step_v(h0,300.0,EU_[k0][0],EU_[k0][1],torch.tensor(SX[rows_[k0],960:992],dtype=torch.float64),torch.stack([EU_[j][0] for j in range(0,k0+1)]),torch.tensor([(B_[k0][0]-B_[j][0])*3600.0 for j in range(0,k0+1)],dtype=torch.float64))))
print('sanity vs SX (k0):', float((base[0][0]-torch.tensor(SX[rows_[k0],192:960].reshape(4,192))).abs().max()))
for nm_,a_ in cands.items():
    pert=run_from(B_,rows_,EU_,k0,k0+4,h0,{k0: torch.tensor(a_*10.0)})
    dy=[float((pert[j][0]-base[j][0]).mean(0).norm()) for j in range(5)]
    do=[float((pert[j][1]-base[j][1]).norm()) for j in range(5)]
    print(nm_,'|dy_mean| per step',np.round(dy,3),' |dout|',np.round(do,4))
```