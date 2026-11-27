The baseline head variant is still the best: view gap = log dt / 10, type 1, register age in minutes, GELU. Every alternative I tried made things much worse, so those choices are settled.

A modest systematic gap remains. The error is 2.0, while about 1.36 would be expected from sampling alone.

Two candidates are left:
- **Prior spread:** the prior's spread may be parameterised differently, e.g. the second half of the prior output as a log-variance or through softplus, rather than a log-sd.
- **Localised error:** the error may be concentrated in one code, or in readings without a previous value of the same code. That would point to a register detail.

I will break the error down by code and register state, and test the two alternative spread forms.

A spread variant that drops the error to about 1.36 would make the head exact. If the gap is instead concentrated in one code or register state, I know exactly what to fix. If neither happens, I will accept the head as accurate enough for design work and use the tool for final checks.

```python
def eval3(opt, sdmode):
    o=[]
    for d in READ:
        prev, dt = d['prev'], d['dt']
        h = prev[192:960].reshape(4,192)*np.exp(-dt*np.exp(Wo['core.flow.a']))
        base = prev[:192] + LNn(h.mean(0),'core.out_norm'); ctx = base + mlp2n(np.array([np.log(dt)/10]),'gap')
        pr = mlp2n(ctx,'z_prior'); ls = pr[32:]
        sd = {'exp':np.exp(ls), 'half':np.exp(0.5*ls), 'softplus':softplus(ls), 'zero':0*ls}[sdmode]
        Z = pr[:32] + sd*ZS
        q = LNn(ctx[None] + linn(Z,'z_head') + Wo['type_emb.weight'][1], 'query_norm')
        has, agemin, zl, vb = d['last']
        wc = q + Wo['embed.code.weight'][CID[d['code']]] + linn(np.array([has,np.log1p(agemin)/10,zl,has,has,0,0,0],float),'last_numeric') + Wo['embed.bins.value.weight'][vb] + Wo['embed.bins.rate.weight'][0]
        lg = linn(gelu(linn(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2') + Wo['embed_transition.value_bin.weight'][CID[d['code']]*17+vb]
        lg -= lg.max(-1,keepdims=True); P = np.exp(lg); P /= P.sum(-1,keepdims=True)
        m_all = P[:,1:]@CENT[d['code']]
        o.append((d['pm'], m_all.mean(), d['nll'], -np.log(P[:,d['ob']].mean()), d['code'], has))
    return pd.DataFrame(o, columns=['pm','mine','nll_t','nll_m','code','has'])
for sdm in ['exp','half','softplus','zero']:
    D = eval3(None, sdm); D['ae'] = (D.pm-D.mine).abs(); D['dn'] = D.nll_t-D.nll_m
    print(sdm, 'MAE %.3f  meanDNLL %.4f |DNLL| %.4f' % (D.ae.mean(), D.dn.mean(), D.dn.abs().mean()))
    if sdm=='exp':
        print(D.groupby(['code','has']).agg(n=('ae','size'), mae=('ae','mean'), dn=('dn','mean'), adn=('dn', lambda x: x.abs().mean())).round(3))
```