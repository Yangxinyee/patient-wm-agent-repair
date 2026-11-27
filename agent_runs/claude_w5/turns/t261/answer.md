ARCHITECTURE.md matches my implementation everywhere except one line: u_k is described as "zero if the bundle has none", while I use the encoder applied to the empty set. That empty-set choice matched the tool at reading bundles, so it can stay.

The useful clue is that z is exact at drug bundles while h is not. That only pins down u if z_post actually depends on u. If z_post is nearly insensitive to u, my action token could be slightly wrong, for example in the rate z-scores, bins or units. The prelude attends almost entirely to u at drug bundles, so that error would show only there, as observed.

So I'll measure z's sensitivity to perturbations of u (and of e for comparison) at j3, and list the numeric normalisation entries available for drug rate and dose.
- If dz/du is tiny, u is unverified and becomes my prime suspect. I'll then test u variants against the tool's j3 state.
- If z is sensitive to u, u is right and I look at the prelude or blocks instead.

```python
for d_ in VOCAB['numeric']:
    print(d_['code'], d_['attribute'], d_['median'], d_['iqr'])
print(sorted(NUMT.keys()))
globals()['WCUR']=W64
z0 = mystep(d0)[2]
g = torch.Generator().manual_seed(0)
for nm in ['u','e']:
    dd = dict(d0); pert = torch.randn(192, generator=g, dtype=torch.float64); pert = pert/pert.norm()*0.01*d0[nm].norm()
    dd[nm] = d0[nm] + pert
    if nm=='e': dd['Mv'] = torch.cat([d0['Mv'][:-1], dd['e'][None]])
    r = mystep(dd)
    print(nm, '|x|', float(d0[nm].norm()), '1% pert -> rel dz', float((r[2]-z0).norm()/z0.norm()), ' rel dh', float((r[1]-mystep(d0)[1]).norm()/mystep(d0)[1].norm()))
print('enc2 code consts:', enc2.__code__.co_consts, 'DRUGS', DRUGS)
print('zs_ names', zs_.__code__.co_names, zs_.__code__.co_consts)
```