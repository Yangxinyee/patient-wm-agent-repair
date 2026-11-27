The kernel has the drug-bundle tensors I need: the prelude output E0, the time-flowed state htf, the target new slots yt and out ot (2,319 drug bundles), plus 520 measurement bundles. I'll check that `step_full` on a subset reproduces yt and ot from E0 and htf, and see what it returns. That makes it usable for computing the response of the state to a perturbation dE.

```python
import torch, time
rw = np.arange(60)
t0=time.time()
res = step_full(rw, E0[rw], htf[rw])
print(type(res), len(res) if isinstance(res,(tuple,list)) else '', time.time()-t0)
outs = res if isinstance(res,(tuple,list)) else (res,)
for o in outs:
    print(getattr(o,'shape',None))
y_, o_ = outs[0], outs[1] if len(outs)>1 else None
print('y err', float((y_-yt[rw]).norm()/yt[rw].norm()), 'out err', float((o_-ot[rw]).norm()/ot[rw].norm()) if o_ is not None and o_.shape==ot[rw].shape else '')
```