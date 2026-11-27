Run 191 confirms my rebuild of a single step: it matches `step_at` exactly (difference 0.0). So I can now propagate my own state through a stay with the true inputs, and simulate weight edits by temporarily changing `Wt`.

The next question is how an injection at an action bundle propagates, since c_out mixes all dimensions. I'll build `run_from`, a multi-step runner that can add a vector to h_mid at chosen bundles (through a temporary b_in bias change). With it I'll measure how the state change after the action bundle shrinks over the next bundles, for three kinds of direction:
- random directions;
- directions whose image after c_out is a low-variance direction of y;
- directions concentrated on fast-decaying (≤17-minute) dimensions.

If some class keeps well under 0.1 of the change one hour later, a non-persistent lag-1 channel is clean. If everything persists unpredictably, the lag structure needs another design.

```python
SPD=torch.tensor(sp_d,dtype=torch.float64)
def run_from(Bs,rws,EUs,k0,k1,h0,inject=None):
    h=h0.clone(); outl=[]
    for k in range(k0,k1+1):
        zk=torch.tensor(SX[rws[k],960:992],dtype=torch.float64)
        dt=(Bs[k][0]-Bs[k-1][0])*3600.0; lo=max(0,k-127)
        Mem_=torch.stack([EUs[j][0] for j in range(lo,k+1)]); Mage_=torch.tensor([(Bs[k][0]-Bs[j][0])*3600.0 for j in range(lo,k+1)],dtype=torch.float64)
        bsave=None
        if inject is not None and k in inject:
            bsave=Wt['core.b_in.bias'].clone(); Wt['core.b_in.bias']+=inject[k]/SPD
        try: h,o=core_step_v(h,dt,EUs[k][0],EUs[k][1],zk,Mem_,Mage_)
        finally:
            if bsave is not None: Wt['core.b_in.bias']=bsave
        outl.append((h.clone(),o.clone()))
    return outl
# pick an insulin action bundle at hour ~14 in first stay with following bundles
ks=[k for k in range(1,len(B_)-4) if any(a['code']==8 and a['kind']==0 for a in B_[k][1]) and B_[k][0]>10]
k0=ks[0]; print('k0',k0,'times',[round(B_[j][0],2) for j in range(k0,k0+5)])
h0=torch.tensor(SX[rows_[k0-1],192:960].reshape(4,192),dtype=torch.float64)
base=run_from(B_,rows_,EUs:=EU_,k0,k0+4,h0)
print('sanity vs SX (k0):', float((base[0][0]-torch.tensor(SX[rows_[k0],192:960].reshape(4,192))).abs().max()))
Ci=np.linalg.pinv(Co)
_,Vy=np.linalg.eigh(np.cov(P_['ym'].T)); lowy=Vy[:,:40]          # 40 lowest-variance mean-y directions
fast=np.argsort(tau)[:10]
cands={}
rng_=np.random.default_rng(0)
for i in range(2): v_=rng_.standard_normal(192); cands['rand%d'%i]=v_/np.linalg.norm(v_)
for i in [0,5,20]: v_=Ci@lowy[:,i]; cands['Cinv_lowy%d'%i]=v_/np.linalg.norm(v_)
v_=np.zeros(192); v_[fast]=1; cands['fast_hmid']=v_/np.linalg.norm(v_)
v_=np.zeros(192); v_[fast]=1; v_=Ci@v_; cands['Cinv_fast_y']=v_/np.linalg.norm(v_)
for nm_,a_ in cands.items():
    inj={k0: torch.tensor(a_*10.0)}
    pert=run_from(B_,rows_,EU_,k0,k0+4,h0,inj)
    dy=[float((pert[j][0]-base[j][0]).mean(0).norm()) for j in range(5)]
    do=[float((pert[j][1]-base[j][1]).norm()) for j in range(5)]
    print(nm_,'|dy_mean| per step',np.round(dy,3),' |dout|',np.round(do,4))
```