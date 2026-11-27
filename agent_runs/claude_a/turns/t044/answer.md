Still no match. The best encoder variant gives an error of 0.077, which is worse than simply using the z prior mean (0.068). So either e_k is wrong in a way I have not searched, or the z_post input differs from what I assumed.

Before searching encoder details any further, I will run some cheap diagnostics:
- e = 0 and u = 0 (does e matter at all?);
- other orders of the input blocks;
- ctx built without the time-flow decay or with a different gap;
- the encoders' activation swapped to ReLU or SiLU.

If one of these drops the error sharply, I will know which part is wrong. If none does, the problem is in the atom embedding itself, and I will need a different way to identify it.

```python
unit_best = {'glucose_finger': 4, 'glucose_lab': 4, 'map_arterial': 2, 'map_cuff': 2}
u2 = torch.tensor(subA2.event.map(unit_best).values)
numf = torch.zeros(len(subA2), 6); numf[:, 0] = zv2t; numf[:, 3] = 1
av_best = P['embed.code.weight'][c2] + P['embed.ekind.weight'][1] + P['embed.categorical.unit.weight'][u2] + P['embed.bins.value.weight'][vb2t] + lin(numf, 'embed.numeric')
def ds_act(av, grp, ngrp, pre, A):
    h1 = lin(A(lin(av, pre+'.mlp1.0')), pre+'.mlp1.2')
    s = torch.zeros(ngrp, 192).index_add_(0, grp, h1); n = torch.zeros(ngrp).index_add_(0, grp, torch.ones(len(grp)))
    s = s / torch.sqrt(n)[:, None] + lin(torch.log1p(n)[:, None], pre+'.size')
    return ln(lin(A(lin(s, pre+'.mlp2.0')), pre+'.mlp2.2'), pre+'.norm')
def zerr(inp, A=F.gelu):
    return round((lin(A(lin(inp, 'z_post.0')), 'z_post.2')[:, :32] - zt2).abs().mean().item(), 4)
e_b = ds_act(av_best, grp2, len(sub_rows), 'encode', F.gelu); Z0 = torch.zeros_like(e_b)
print('ctx,e,0', zerr(torch.cat([ctxs, e_b, Z0], 1)), ' ctx,0,0', zerr(torch.cat([ctxs, Z0, Z0], 1)), ' e,ctx,0', zerr(torch.cat([e_b, ctxs, Z0], 1)), ' ctx,0,e', zerr(torch.cat([ctxs, Z0, e_b], 1)))
# ctx variants
pv2 = torch.tensor(SB.prev.values[sub_rows].astype(np.int64)); dts2 = torch.tensor(SB.dts.values[sub_rows], dtype=torch.float32)
c_nodecay = OUT[pv2] + ln(HS[pv2].mean(1), 'core.out_norm') + mlp((torch.log(dts2)/10)[:, None], 'gap', F.gelu)
c_nogap = OUT[pv2] + ln((HS[pv2]*torch.exp(-dts2[:, None, None]*torch.exp(P['core.flow.a']))).mean(1), 'core.out_norm')
c_gap1 = c_nogap + mlp(torch.log(dts2)[:, None], 'gap', F.gelu)
for nm_, cc_ in [('nodecay', c_nodecay), ('nogap', c_nogap), ('gap_nodiv', c_gap1), ('out only', OUT[pv2])]:
    print(nm_, zerr(torch.cat([cc_, e_b, Z0], 1)))
for an, A in [('relu', F.relu), ('silu', F.silu)]:
    e_a = ds_act(av_best, grp2, len(sub_rows), 'encode', A)
    print('encoder act', an, zerr(torch.cat([ctxs, e_a, Z0], 1)), 'zpost act also', zerr(torch.cat([ctxs, e_a, Z0], 1), A))
```