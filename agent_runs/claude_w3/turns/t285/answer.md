The best setting (type 1, gap of log dt/10, layout [has, log-age/10, 6 numerics], age in hours) gives a mean absolute error of 2.45, which is close but not exact. Next I'll test the remaining variants in one run: age in minutes, renormalising p over bins 1–16, the rate-bin and status indices, and more z samples. The per-code error breakdown will show which code is off.

```python
def head_mean2(type_i=1, au=3600, renorm=False, rb=0, ls=0, P=P0, extra=None, zh_=None, onv=None, outv=None):
    zz = zh if zh_ is None else zh_
    view = ((out0 if outv is None else outv) + (on0 if onv is None else onv) + gapf(np.log(dtv)*0.1))[:, None, :] + zz
    q = LNn(view + P['type_emb.weight'][type_i], P['query_norm.weight'], P['query_norm.bias'])
    a = np.where(has, np.log1p(np.nan_to_num(agev)/au)/10, 0)
    feat = np.stack([has*1.0, a, zl_, has*1.0, 0*a, 0*a, 0*a, 0*a], 1)
    base = P['embed.code.weight'][cv] + li(feat, 'last_numeric', P) + P['embed.bins.value.weight'][lvb]
    if rb is not None: base = base + P['embed.bins.rate.weight'][rb]
    if ls is not None: base = base + P['last_status.weight'][ls]
    wc = q + base[:, None, :] + (0 if extra is None else extra)
    lg = li(gel(li(wc, 'attr_heads.value_bin.0', P)), 'attr_heads.value_bin.2', P) + P['embed_transition.value_bin.weight'][cv*17 + lvb][:, None, :]
    p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); p = p.mean(1)
    pp = p[:, 1:] / p[:, 1:].sum(1, keepdims=True) if renorm else p[:, 1:]
    return (pp * cent).sum(1), p
for nm, kw in [('base', {}), ('min', {'au': 60}), ('renorm', {'renorm': True}), ('rbNone', {'rb': None}), ('ls1', {'ls': 1}), ('lsNone', {'ls': None}), ('day', {'au': 86400})]:
    m_, p_ = head_mean2(**kw); e = np.abs(m_ - tool)
    print(nm, e.mean().round(3), {c: e[cv == c].mean().round(2) for c in [5, 6, 7, 10]}, 'p0', p_[:, 0].mean().round(4))
```