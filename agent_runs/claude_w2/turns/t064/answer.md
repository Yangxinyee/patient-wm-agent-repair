The grid gives no clean solution. Every combination of decay scale and gap unit still leaves about 3 mg/dL of error, with biases that differ by code. The parameters trade off against each other, so my assumed structure of the view is probably off rather than a unit.

Next I test structural variants of the view:
- dropping the `out` term;
- dropping the decayed-slots term;
- doubling `out`;
- dropping z.

I also compare my bin probabilities with the tool's per-reading NLL. That shows whether my distributions are shifted or wrongly shaped. A variant that brings all four codes' biases below 1 mg/dL and its NLL in line with the tool's would settle the structure.

```python
def run7(f8, vmode='full', fs=1.0, ty=1):
    decf = torch.exp(-(dt_*fs)[:,None] * torch.exp(Wd['core.flow.a'])[None,:])
    hn = LN_((HH[P_]*decf[:,None,:]).mean(1), 'core.out_norm'); o_ = OUT[P_]
    g_ = M((torch.log(dt_)/10)[:,None], 'gap')
    parts = {'full': o_+hn+g_, 'no_out': hn+g_, 'no_hn': o_+g_, 'two_out': 2*o_+g_, 'no_gap': o_+hn}
    ctx_ = parts[vmode]
    zmu = M(ctx_, 'z_prior')[:, :32]
    res = {}
    for zm in ['z','noz']:
        v_ = ctx_ + (L(zmu, 'z_head') if zm=='z' else 0)
        q_ = LN_(v_ + Wd['type_emb.weight'][ty], 'query_norm')
        wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t]
        lg = M(wc, 'attr_heads.value_bin') + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
        p_ = torch.softmax(lg, -1).numpy()
        pm7 = (p_[:,1:]*cenmat).sum(1)
        ob = np.array([fb(e_, v2) for e_, v2 in zip(fa4.event, fa4.observed)])
        nll_m = -np.log(p_[np.arange(len(p_)), ob] + 1e-12)
        res[zm] = (rep(pm7), round(float(np.mean(nll_m)),4), round(float(np.mean(fa4.nll)),4))
    return res
for vm in ['full','no_out','no_hn','two_out','no_gap']:
    print(vm, run7(fC, vm))
```