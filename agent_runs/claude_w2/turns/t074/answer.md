The NE pattern was a side effect, not a missing NE input. After controlling for the last value's z, NE barely matters. The last value alone explains about 76% of my MAP error: the real model follows the last reading more closely than my head does. Finger glucose shows the same pattern, more weakly. Rescaling the z input does not help, so the shortfall must come from the bin-based terms (the transition table and the last-bin embedding).

The likely culprit is bin assignment. Readings and bin edges are both integers, so many readings sit exactly on an edge. My `fb` assigns those to the upper bin (right-sided search), whereas my earlier `zpred` defaulted to left-sided. A one-bin shift for every on-edge reading would blunt persistence and also distort my NLL.

Test: recompute the last-value bins and the observed bins with left-sided search. If this is the cause, error and NLL should drop sharply for all codes.

```python
def fbl(code, x): return 1 + np.searchsorted(np.array(num_[(code,'value')]['fine_bin_edges']), x, side='left')
lbL = torch.tensor(np.array([fbl(e_, v_) if not np.isnan(v_) else 0 for e_, v_ in zip(fa4.event, fa4['last'])]))
obL = np.array([fbl(e_, v2) for e_, v2 in zip(fa4.event, fa4.observed)])
def run13(f8, lbt, obb, a='gelu', ty=1):
    dec_ = torch.exp(-dt_[:,None]*torch.exp(Wd['core.flow.a'])[None,:])
    b_ = OUT[P_] + LN_((HH[P_]*dec_[:,None,:]).mean(1), 'core.out_norm')
    ctx_ = b_ + Ma((torch.log(dt_)/10)[:,None], 'gap', a)
    zmu = Ma(ctx_, 'z_prior', a)[:, :32]
    q_ = LN_(ctx_ + L(zmu, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lbt]
    lg = Ma(wc, 'attr_heads.value_bin', a) + Wd['embed_transition.value_bin.weight'][codes_t*17 + lbt]
    p_ = torch.softmax(lg, -1).numpy(); pm_ = (p_[:,1:]*cenmat).sum(1)
    nl_ = -np.log(p_[np.arange(len(p_)), obb] + 1e-12)
    return pm_, p_, nl_, wc, q_, lg
for nmv, lbt, obb in [('right/right', lb_t, ob_), ('left/left', lbL, obL), ('left last/right obs', lbL, ob_)]:
    pmv, _, nlv, _, _, _ = run13(f8_(), lbt, obb)
    print(nmv, rep2(pmv, nlv), fa4.assign(ad=pmv-tgt, n_=nlv).groupby('event')[['ad','n_']].mean().round(3).to_dict())
for au in [60.0, 3600.0]:
    pmv, _, nlv, _, _, _ = run13(f8_(au), lbL, obL); print('left, age unit', au, rep2(pmv, nlv))
```