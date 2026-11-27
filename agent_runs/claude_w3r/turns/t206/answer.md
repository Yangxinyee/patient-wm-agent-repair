Run 193 fixed the unpacking (`core_step_v` returns 3 values), but `cands` was never created because run 192 crashed before that line. The sanity gap of 7.0 against the stored state is my local implementation's known inexactness: about 4.5% relative error in h, with entries of size ~100–200. Since I only compare perturbed against unperturbed runs of the same implementation, it doesn't affect the test.

Now I'll define the candidate directions and run the persistence test. I expect fast-decaying y directions to vanish by the next hour, while random directions persist or spread. That is what decides whether a non-persistent lag-1 channel is feasible.

```python
Ci=np.linalg.pinv(Co)
_,Vy=np.linalg.eigh(np.cov(P_['ym'].T)); lowy=Vy[:,:40]
fast=np.argsort(tau)[:10]
cands={}; rng_=np.random.default_rng(0)
for i in range(2): v_=rng_.standard_normal(192); cands['rand%d'%i]=v_/np.linalg.norm(v_)
for i in [0,5,20]: v_=Ci@lowy[:,i]; cands['Cinv_lowy%d'%i]=v_/np.linalg.norm(v_)
v_=np.zeros(192); v_[fast]=1; cands['fast_hmid']=v_/np.linalg.norm(v_)
v_=np.zeros(192); v_[fast]=1; v_=Ci@v_; cands['Cinv_fast_y']=v_/np.linalg.norm(v_)
for nm_,a_ in cands.items():
    pert=run_from(B_,rows_,EU_,k0,k0+4,h0,{k0: torch.tensor(a_*10.0)})
    dy=[float((pert[j][0]-base[j][0]).mean(0).norm()) for j in range(5)]
    do=[float((pert[j][1]-base[j][1]).norm()) for j in range(5)]
    print(nm_,'|dy_mean| per step',np.round(dy,3),' |dout|',np.round(do,4))
```