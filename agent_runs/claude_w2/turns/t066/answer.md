Using the reading's own posterior z gives an NLL of 1.301, better than the tool's 1.4235. That variant sees too much. The prior gives 1.567, too little. So the tool most likely uses the prior, and my prior or view computation is slightly wrong.

Two leads from my earlier work:
- My earlier z-posterior match used gap(log dt) without the /10, while the view fits best with /10. The prior's context and the view may therefore use different gap inputs.
- My context function referenced GELU, so the activation may differ between modules (gap MLP, z prior, value head).

Next I search over:
- the gap form for the prior context;
- the activation of each of the three MLPs (SiLU, GELU, tanh-GELU).

I score by both mean absolute error and NLL against the tool. A match should bring my NLL to about 1.42 and the error well below 2.

```python
AF = {'silu': F.silu, 'gelu': F.gelu, 'gelut': lambda x: F.gelu(x, approximate='tanh')}
def Ma(x, pre, a='silu'): return L(AF[a](L(x, pre+'.0')), pre+'.2')
def run9(f8, cg='log10', vg='log10', a_gap='silu', a_zp='silu', a_vh='silu', ty=1, ret=False):
    hn = LN_(hfl.mean(1), 'core.out_norm'); b_ = OUT[P_] + hn
    gin = {'log10': torch.log(dt_)/10, 'log': torch.log(dt_)}
    ctx_ = b_ + Ma(gin[cg][:,None], 'gap', a_gap); view_ = b_ + Ma(gin[vg][:,None], 'gap', a_gap)
    zmu = Ma(ctx_, 'z_prior', a_zp)[:, :32]
    q_ = LN_(view_ + L(zmu, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t]
    lg = Ma(wc, 'attr_heads.value_bin', a_vh) + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    p_ = torch.softmax(lg, -1).numpy()
    pm9 = (p_[:,1:]*cenmat).sum(1); nll_m = -np.log(p_[np.arange(len(p_)), ob_] + 1e-12)
    if ret: return pm9, p_, nll_m, wc, q_
    return (rep(pm9), round(float(nll_m.mean()),4))
o9 = []
for cg, vg, ag, az, av in itertools.product(['log10','log'], ['log10'], ['silu','gelu'], ['silu','gelu'], ['silu','gelu']):
    o9.append((run9(fC, cg, vg, ag, az, av), cg, vg, ag, az, av))
o9.sort(key=lambda t: t[0][0][0])
for o_ in o9[:8]: print(o_)
```