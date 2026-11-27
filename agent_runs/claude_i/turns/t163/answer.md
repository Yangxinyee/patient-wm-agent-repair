The driver reproduces the tool's states when run from scratch: stay 0 has median relative error 1e-4. The other stays came out NaN because `dts` for a stay's first bundle is the negative gap to the previous stay (−97,800 s). The first bundle should use dt = 0, as stay 0 does.

Stay 0 also shows a maximum error of 10% at some bundles. I need to know whether those are isolated (a few bundle types that `core_step` handles slightly differently) or whether errors grow along the stay.

I am fixing the first-bundle dt and running all 100 stays. I'll summarise the error and relate the worst bundles to their contents (insulin/NE flags, dt) using the tool's states file. If the large errors are rare and tied to one kind of bundle, the simulator is good enough for designing edits. If they accumulate, I must find the discrepancy first.

```python
code_sim = open('/work/sim.py').read().replace("hf = h * torch.exp(-dts[k] * torch.exp(WT['core.flow.a']))\n        ctx = ctx_from(o, hf, dts[k])",
 "dtk = dts[k] if k!=j0 else torch.zeros((), dtype=dts.dtype)\n        hf = h * torch.exp(-dtk * torch.exp(WT['core.flow.a']))\n        ctx = ctx_from(o, hf, dtk)")
open('/work/sim.py','w').write(code_sim); exec(code_sim)
import numpy as np, torch
z3_=np.load([f for f in ['/work/fwd_3/states_original.npz'] ][0], allow_pickle=True)
print(z3_.files)
stay_ks={}
for i_,s_ in enumerate(starts):
    e_=starts[i_+1] if i_+1<len(starts) else len(first_of); stay_ks[s_]=list(range(s_,e_))
ALLerr=np.zeros(len(first_of)); ALLz=np.zeros(len(first_of))
for s_,ks in stay_ks.items():
    Hs,Os,Zs,CT=run_stay(ks, Ee, Uu, collect=True)
    for i_,k in enumerate(ks):
        T_=torch.as_tensor(HHt[k]).to(Hs[i_].dtype)
        ALLerr[k]=float((Hs[i_]-T_).norm()/T_.norm())
        ALLz[k]=float((Zs[i_]-torch.as_tensor(ZZt[k]).to(Zs[i_].dtype)).norm()/(torch.as_tensor(ZZt[k]).norm()+1e-9))
print("h relerr median %.2e 90%% %.2e 99%% %.2e max %.2e"%tuple(np.quantile(ALLerr,[.5,.9,.99,1])))
print("z relerr median %.2e 90%% %.2e 99%% %.2e max %.2e"%tuple(np.quantile(ALLz,[.5,.9,.99,1])))
bad=np.where(ALLerr>0.01)[0]; print("n bad",len(bad))
for f_ in z3_.files:
    a_=z3_[f_]
    if a_.shape[0]==len(first_of) and a_.ndim==1: print(f_, "bad mean", a_[bad].astype(float).mean() if a_.dtype!=object else '', "all mean", a_.astype(float).mean() if a_.dtype!=object else '')
print("dts bad median", float(np.median(dts.numpy()[bad])), "all median", float(np.median(dts.numpy())))
print("U nonzero frac bad", float((Uu[bad].abs().sum(-1)>0).float().mean()), "all", float((Uu.abs().sum(-1)>0).float().mean()))
```