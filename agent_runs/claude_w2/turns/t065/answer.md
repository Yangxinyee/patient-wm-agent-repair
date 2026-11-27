My re-implementation is clearly missing information. Its bin NLL is 1.567 against the tool's 1.4235, and the tool's predictions track the observed values more closely than mine.

A likely explanation: in factual (teacher-forced) mode the head uses the z posterior of the bundle being predicted, which has seen that bundle, rather than the prior. That would make the predictions sharper and pull them toward the observed values, which is exactly what I see.

Test: use the stored posterior mean of the reading's own bundle in the view, and compare error and NLL.
- If this closes most of the 0.14-nat gap, the factual metric is posterior-based.
- That would mean the factual-fit check and the dose-edit rollouts, which must use the prior for hidden readings, see different z's. The readout will have to work in both.

```python
ZP = X_[:, 960:992]
R_ = torch.tensor(fa4.row.values)
ob_ = np.array([fb(e_, v2) for e_, v2 in zip(fa4.event, fa4.observed)])
def run8(f8, zsrc='post_cur', fs=1.0, gdiv=10.0, ty=1):
    decf = torch.exp(-(dt_*fs)[:,None] * torch.exp(Wd['core.flow.a'])[None,:])
    hn = LN_((HH[P_]*decf[:,None,:]).mean(1), 'core.out_norm')
    ctx_ = OUT[P_] + hn + M((torch.log(dt_)/gdiv)[:,None], 'gap')
    z_ = {'post_cur': ZP[R_], 'post_prev': ZP[P_], 'prior': M(ctx_, 'z_prior')[:, :32]}[zsrc]
    q_ = LN_(ctx_ + L(z_, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t]
    lg = M(wc, 'attr_heads.value_bin') + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    p_ = torch.softmax(lg, -1).numpy()
    pm8 = (p_[:,1:]*cenmat).sum(1)
    nll_m = -np.log(p_[np.arange(len(p_)), ob_] + 1e-12)
    return pm8, p_, nll_m, wc
for zs_ in ['post_cur','post_prev','prior']:
    for gd in [10.0, 1.0]:
        pm8, p8, nl8, _ = run8(fC, zs_, gdiv=gd)
        print(zs_, gd, rep(pm8), 'nll mine', round(float(nl8.mean()),4), 'tool', round(float(fa4.nll.mean()),4), 'corr nll', round(np.corrcoef(nl8, fa4.nll)[0,1],3))
```