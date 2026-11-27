`e_all` is no longer the per-bundle event-token table the replica's memory expects. A later run overwrote it with a 76,960-length vector, and the original 5,218×192 table is gone. I can rebuild most of it. The 2,319 drug bundles have exact event tokens in `TOK[:,4]`. For the other bundles I have only a sample (`TOKn[:,4]`, 520 bundles of unknown position), so for now I'll fill them with that sample's mean.

This should be enough for my purpose. I need the Jacobian of the new slots and `out` with respect to an injection, not the exact level. Most of an injection's effect should flow through the linear state-update path, decay·h + softplus-step·b_in·E, which does not touch the memory at all.

Next I'll:
1. Back up the broken `e_all` and install the rebuilt table as a torch tensor.
2. Run `step_full` on 64 drug bundles and compare with the recorded slots and `out`, and time it.

A good fit, or a moderate one that the memory approximation could explain, lets me go on to the Jacobian. In the next run I'll also check how sensitive that Jacobian is to the memory approximation.

```python
import time
e_all_old=e_all
TOKt=TOK if isinstance(TOK,torch.Tensor) else torch.tensor(TOK,dtype=torch.float32)
TOKnt=TOKn if isinstance(TOKn,torch.Tensor) else torch.tensor(TOKn,dtype=torch.float32)
eA=TOKnt[:,4].mean(0,keepdim=True).repeat(len(sid),1).clone()
eA[torch.as_tensor(di)]=TOKt[:,4]
e_all=eA
print('drug e-token norm %.2f, non-drug mean e norm %.2f'%(TOKt[:,4].norm(dim=1).mean(),TOKnt[:,4].mean(0).norm()))
t0=time.time()
with torch.no_grad():
    E0t=E0 if isinstance(E0,torch.Tensor) else torch.tensor(E0,dtype=torch.float32)
    htft=htf if isinstance(htf,torch.Tensor) else torch.tensor(htf,dtype=torch.float32)
    res=step_full(di[:64], E0t[:64], htft[:64])
print('time %.2fs'%(time.time()-t0), type(res).__name__, [tuple(getattr(r,'shape',())) for r in (res if isinstance(res,(tuple,list)) else [res])])
outs=res if isinstance(res,(tuple,list)) else [res]
yt_=np.asarray(yt[:64]); ot_=np.asarray(ot[:64])
for r in outs:
    r=np.asarray(r)
    if r.shape==yt_.shape: e=r-yt_; print('y: rms err %.3f  R2 %.5f'%(np.sqrt((e**2).mean()), 1-(e**2).sum()/((yt_-yt_.mean(0))**2).sum()))
    if r.shape==ot_.shape: e=r-ot_; print('out: rms err %.4f R2 %.5f'%(np.sqrt((e**2).mean()), 1-(e**2).sum()/((ot_-ot_.mean(0))**2).sum()))
```