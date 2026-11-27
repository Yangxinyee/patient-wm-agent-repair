Matching worked. The best variant reproduces the tool's predicted means with correlation 0.9989 and mean absolute error 3.8, across all four codes. That variant uses:
- gap(log dt / 10);
- type embedding 1;
- age in hours;
- `last_numeric` inputs [has_last, log(1+age)/10, z_last, has_last, 0, 0, 0, 0].

`last_status` makes no difference.

An error of 3.8 is still too large for calibrating a readout, so I refine:
- all 15 type indices;
- a separate gap scaling for the z-prior context and for the view;
- which other `last_numeric` columns carry the has_last flag.

Some error should remain, since I use the prior mean of z instead of 4 random draws. I will also report the error by code so I can see where the mismatch sits.

```python
def run2(gctx='log10', gview='log10', ty=1, hascols=(0,3), age_unit=3600.0, zmode='mean', status=None):
    ctx_ = base0 + gapv(gctx)
    view_b = base0 + gapv(gview)
    zmu = M(ctx_, 'z_prior')[:, :32]
    view = view_b + (L(zmu, 'z_head') if zmode=='mean' else 0)
    q_ = LN_(view + Wd['type_emb.weight'][ty], 'query_norm')
    la = np.log1p(fa4.age_s.values/age_unit)/10 * fa4.has.values
    f = np.zeros((len(fa4), 8)); f[:,1]=la; f[:,2]=fa4.zl
    for c_ in hascols: f[:,c_]=fa4.has
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t] + Wd['embed.bins.rate.weight'][0]
    if status is not None: wc = wc + Wd['last_status.weight'][status]
    lg = M(wc, 'attr_heads.value_bin') + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    p_ = torch.softmax(lg, -1).numpy()
    return (p_[:,1:]*cenmat).sum(1), p_, wc
def sc(pm_): return round(float(np.mean(np.abs(pm_-tgt))),3)
r2_ = []
for ty in range(15): r2_.append((sc(run2(ty=ty)[0]), 'ty', ty))
for gc, gv in [('log','log10'),('log10','log'),('log','log')]: r2_.append((sc(run2(gctx=gc, gview=gv)[0]), 'gap', gc, gv))
for hc in [(3,),(0,),(0,3,5),(0,3,7),(0,3,6),(0,3,4),(0,1,3),(0,3,5,7)]: r2_.append((sc(run2(hascols=hc)[0]), 'has', hc))
r2_.append((sc(run2(zmode='none')[0]), 'noz'))
r2_.sort(key=lambda t: t[0]); print(r2_[:12])
pm_, pp_, _ = run2()
fa4['mine'] = pm_
print(fa4.assign(ad=np.abs(fa4.mine-fa4.predicted_mean)).groupby('event').ad.describe().round(2))
print(Wd['last_status.weight'].norm(dim=1))
```