Even with the simplest bundles, nothing matches: the best error is 0.041 against a scale of 0.105. Notably, the error hardly changes with the unit, kind, event-kind or kq choices. So the atom embedding is not the main problem; the error comes from ctx or from how the z_post output is read.

Next I will test a few cheap alternatives on the single-atom set, keeping the best encoder settings fixed:
- the mean taken from the second half of the z_post output;
- a ctx built from out only, or from the normalised h only;
- the gap computed in hours rather than seconds.

If one of these drops the error close to zero, I have found the convention. If none does, I will stop rebuilding the core and switch to measurement-based design using the forward tool's states.

```python
nf = torch.zeros(len(sub1), 6); nf[:, 0] = zv1t; nf[:, 4] = 1
av = P['embed.code.weight'][c1] + P['embed.bins.value.weight'][torch.tensor(vb1)] + lin(nf, 'embed.numeric') + P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + P['embed.kq.weight'][0]
e1 = ln(mlp(mlp(av, 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm')
Z1 = torch.zeros_like(e1)
hd1 = (HS[pv1]*torch.exp(-dt1[:, None, None]*torch.exp(P['core.flow.a'])))
variants = {
 'nd': ctx1['nd'], 'd': ctx1['d'],
 'out+g': OUT[pv1] + g1, 'lnh_nd+g': ln(HS[pv1].mean(1), 'core.out_norm') + g1, 'lnh_d+g': ln(hd1.mean(1), 'core.out_norm') + g1,
 'nd_ghours': OUT[pv1] + ln(HS[pv1].mean(1), 'core.out_norm') + mlp((torch.log(dt1/3600)/10)[:, None], 'gap', F.gelu),
 'nd_nogap': OUT[pv1] + ln(HS[pv1].mean(1), 'core.out_norm'),
 'out_only': OUT[pv1],
}
for nm_, cc_ in variants.items():
    o_ = mlp(torch.cat([cc_, e1, Z1], 1), 'z_post', F.gelu)
    o0 = mlp(torch.cat([cc_, Z1, Z1], 1), 'z_post', F.gelu)
    print(nm_, 'first32', round((o_[:, :32] - z1).abs().mean().item(), 4), 'last32', round((o_[:, 32:] - z1).abs().mean().item(), 4), ' e=0 first32', round((o0[:, :32] - z1).abs().mean().item(), 4))
# correlation structure: how well can z1 be linearly predicted from ctx_nd vs ctx_d (diagnostic)
from sklearn.linear_model import RidgeCV
for nm_ in ['nd', 'd']:
    Xd = torch.cat([variants[nm_], e1], 1).numpy(); yd = z1.numpy()
    pr_ = np.zeros_like(yd)
    for trn, tst in GroupKFold(5).split(Xd, yd, SB.stay_id.values[n1_rows]):
        pr_[tst] = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(Xd[trn], yd[trn]).predict(Xd[tst])
    print('linear predictability of z from', nm_, 'ctx+e: mae', np.abs(pr_ - yd).mean().round(4))
```