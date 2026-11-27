`bvd(ev)` returns the bundle's (e_k, u_k) pair, and `st0` is just an index.

Next I will call `step` on the first bundle of the first stay, from a zero state and with the tool's z. That shows its return structure. I will also check that the tool's state rows line up one-to-one with my bundle list for that stay. Both are prerequisites for a full recursion test.

```python
import torch, math
m0=S['stay_id']==s0
print('rows',m0.sum(),'bundles',len(BUN[s0]), 'Wt dtype', Wt['core.flow.a'].dtype, tuple(Wt['gap.0.weight'].shape))
xs=torch.tensor(S['x'][m0],dtype=Wt['core.flow.a'].dtype)
e,u=bvd(BUN[s0][0][1])
r=step(torch.zeros(192,dtype=xs.dtype), torch.zeros(4,192,dtype=xs.dtype), 0.0, e.to(xs.dtype), u.to(xs.dtype), xs[0,-32:], torch.zeros(0,192,dtype=xs.dtype), torch.zeros(0,dtype=xs.dtype))
print(type(r).__name__, [tuple(t.shape) if hasattr(t,'shape') else type(t).__name__ for t in r])
print('out err', float((r[0]-xs[0,:192]).abs().max()))
```