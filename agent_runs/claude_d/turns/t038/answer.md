The structure is nearly right. My first-32 z_post output correlates 0.94 with the target, so the layout (mean first) is right, but the values are not exact. Other observations:
- out has norm 13.9 ≈ √192, which fits a unit-weight LayerNorm output.
- The h slots have very large norms (about 467).
- After bundle 1 the posterior means are much smaller (about 0.1).

The gap input for the first bundle (dt undefined) and the initial context are the likeliest culprits. Next I will scan the gap-MLP input over a range of values. I will also try the initial context with and without the `out_norm` bias, and a few small embedding variants (kq row on/off, LayerNorm eps). An exact match at some setting pins down the first-bundle convention; if nothing gets close, the atom embedding itself is wrong.

```python
def first_err(gv, use_norm=True, kq=True, eps=1e-5, act=F.gelu, extra=None):
    a = atom_vec('hospital_admission', 0, 3)
    if not kq: a = a - T['embed.kq.weight'][0]
    if extra is not None: a = a + extra
    e1 = set_enc([a], 'encode', act)
    ctx = (ln(torch.zeros(192),'core.out_norm') if use_norm else torch.zeros(192)) + (mlp(torch.tensor([gv],dtype=torch.float32),'gap',act) if gv is not None else 0)
    zp = mlp(torch.cat([ctx, e1, torch.zeros(192)]), 'z_post', act)
    return (zp[:32]-target_z1).abs().max().item(), zp
best=[]
for gv in list(np.linspace(-2,2,41))+[None]:
    for un in [True, False]:
        for kq in [True, False]:
            e,_ = first_err(gv, un, kq); best.append((e,gv,un,kq))
best.sort(key=lambda x:x[0]); print(best[:8])
# check e1 alternatives: maybe visits have unit/rate_unit rows or kind/ekind differ
for ek in range(5):
    for kd in [0,2]:
        a = atom_vec('hospital_admission', kd, ek)
        e1 = set_enc([a],'encode',F.gelu)
        ctx = ln(torch.zeros(192),'core.out_norm') + mlp(torch.tensor([0.0]),'gap',F.gelu)
        zp = mlp(torch.cat([ctx, e1, torch.zeros(192)]), 'z_post', F.gelu)
        print('ekind',ek,'kind',kd, round((zp[:32]-target_z1).abs().max().item(),3))
```