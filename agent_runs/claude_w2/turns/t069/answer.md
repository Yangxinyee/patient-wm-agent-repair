Sharing the register between paired codes makes the fit worse. Each code keeps its own register, as documented.

That leaves smaller unknowns. My earlier choices of age unit, gap scale and `last_numeric` layout were made with SiLU, and switching to GELU may change which is best.

Next I re-run those searches with GELU, adding two candidate extra terms in the head input: the event-kind embedding for measurements, and the WORLD atom-kind embedding. I also break my NLL gap down by code. If one code carries most of the 0.06-nat gap, that points to a code-specific missing input.

```python
def run12(f8, gfun=lambda d: torch.log(d)/10, a='gelu', add=None, ty=1):
    dec_ = torch.exp(-dt_[:,None]*torch.exp(Wd['core.flow.a'])[None,:])
    b_ = OUT[P_] + LN_((HH[P_]*dec_[:,None,:]).mean(1), 'core.out_norm')
    ctx_ = b_ + Ma(gfun(dt_)[:,None], 'gap', a)
    zmu = Ma(ctx_, 'z_prior', a)[:, :32]
    q_ = LN_(ctx_ + L(zmu, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t]
    if add is not None: wc = wc + add
    lg = Ma(wc, 'attr_heads.value_bin', a) + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    p_ = torch.softmax(lg, -1).numpy(); pm_ = (p_[:,1:]*cenmat).sum(1)
    nl_ = -np.log(p_[np.arange(len(p_)), ob_] + 1e-12)
    return pm_, p_, nl_, wc, q_
# NOTE: fa4 was re-sorted in run 63; rebuild aligned tensors
P_ = torch.tensor(fa4.prev.values); dt_ = torch.tensor(fa4.dt.values); codes_t = torch.tensor(fa4.code.values)
fa4['has'] = fa4['last'].notna().astype(float)
fa4['zl'] = [zs(e_, v_) if not np.isnan(v_) else 0.0 for e_, v_ in zip(fa4.event, fa4['last'])]
fa4['lb'] = [fb(e_, v_) if not np.isnan(v_) else 0 for e_, v_ in zip(fa4.event, fa4['last'])]
fa4['age_s'] = ((fa4.hour - fa4.last_h)*3600).fillna(0)
lb_t = torch.tensor(fa4.lb.values); cenmat = np.stack([cen[e_] for e_ in fa4.event]); tgt = fa4.predicted_mean.values
ob_ = np.array([fb(e_, v2) for e_, v2 in zip(fa4.event, fa4.observed)])
def f8_(age_unit=3600.0, hascols=(0,3)):
    la = np.log1p(fa4.age_s.values/age_unit)/10*fa4.has.values
    f = np.zeros((len(fa4),8)); f[:,1]=la; f[:,2]=fa4.zl.values
    for c_ in hascols: f[:,c_]=fa4.has.values
    return f
def rep2(pm_, nl_): return (round(float(np.abs(pm_-tgt).mean()),3), round(float(nl_.mean()),4))
out12 = []
for au, hc, gname in itertools.product([1.0, 60.0, 3600.0], [(0,3),(0,),(0,3,5),(0,3,7)], ['log10','logmin10','log1p10']):
    gf_ = {'log10': lambda d: torch.log(d)/10, 'logmin10': lambda d: torch.log(d/60)/10, 'log1p10': lambda d: torch.log1p(d)/10}[gname]
    pm_, _, nl_, _, _ = run12(f8_(au, hc), gf_)
    out12.append((rep2(pm_, nl_), au, hc, gname))
out12.sort(key=lambda t: t[0][1]); print(out12[:6])
for nmA, addv in [('ekind1', Wd['embed.ekind.weight'][1]), ('kind0', Wd['embed.kind.weight'][0]), ('both', Wd['embed.ekind.weight'][1]+Wd['embed.kind.weight'][0])]:
    pm_, _, nl_, _, _ = run12(f8_(), add=addv); print(nmA, rep2(pm_, nl_))
pm_, _, nl_, _, _ = run12(f8_())
print(fa4.assign(mine=nl_).groupby('event')[['mine','nll']].mean().round(3))
```