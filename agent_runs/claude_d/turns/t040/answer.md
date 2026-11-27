The documented order [ctx, e_k, u_k] is clearly best (correlation 0.94, against at most 0.89 for the others). So the remaining error lies inside ctx or the bundle encoding. The most uncertain piece is the initial state: h might start at the learned `core.slots` rather than zero, and out might start as `out_norm` of the mean slot.

Next I will try those initial-context variants combined with a scan over the first bundle's gap input. If one combination brings the error near zero, I have the initialisation. If not, the problem is in the hospital-admission atom's encoding.

```python
slots = T['core.slots']
on0 = ln(torch.zeros(192),'core.out_norm'); onS = ln(slots.mean(0),'core.out_norm')
variants = {'bias+gap': on0, 'normslots+gap': onS, '2normslots': 2*onS, 'gaponly': torch.zeros(192), 'normslots+bias': onS+on0}
res=[]
for vn, base in variants.items():
    for gv in list(np.linspace(-1.5,1.5,31))+[None]:
        ctx = base + (mlp(torch.tensor([gv],dtype=torch.float32),'gap',F.gelu) if gv is not None else 0)
        zp = mlp(torch.cat([ctx, e1, torch.zeros(192)]), 'z_post', F.gelu)
        res.append(((zp[:32]-target_z1).abs().max().item(), vn, gv))
res.sort(key=lambda r:r[0]); print(res[:10])
```