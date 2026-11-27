Type 1 fits every code best. The age units, though, favour minutes for the two MAP codes and hours for the two glucose codes, which suggests a code-dependent offset from a term I have slightly wrong. Next I'll vary the gap inputs (view and context) and the log-age form.

```python
def run_var(gv, gc, au, zh_=zh64):
    global ctx0
    ctx_ = out0 + on0 + gapf(gc)
    zp_ = li(gel(li(ctx_, 'z_prior.0', P0)), 'z_prior.2', P0)
    r_ = np.random.default_rng(3); zs_ = zp_[:, None, :32] + np.exp(zp_[:, 32:])[:, None, :] * r_.standard_normal((len(R), 64, 32))
    zz = li(zs_, 'z_head', P0)
    view = (out0 + on0 + gapf(gv))[:, None, :] + zz
    q = LNn(view + P0['type_emb.weight'][1], P0['query_norm.weight'], P0['query_norm.bias'])
    a = np.where(has, np.log1p(np.nan_to_num(agev)/au)/10, 0)
    feat = np.stack([has*1.0, a, zl_, has*1.0, 0*a, 0*a, 0*a, 0*a], 1)
    wc = q + (P0['embed.code.weight'][cv] + li(feat, 'last_numeric', P0) + P0['embed.bins.value.weight'][lvb])[:, None, :]
    lg = li(gel(li(wc, 'attr_heads.value_bin.0', P0)), 'attr_heads.value_bin.2', P0) + P0['embed_transition.value_bin.weight'][cv*17 + lvb][:, None, :]
    p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); p = p.mean(1)
    m_ = (p[:, 1:] * cent).sum(1); e = np.abs(m_ - tool)
    return e.mean().round(3), [e[cv == c].mean().round(2) for c in [5, 6, 7, 10]]
L10 = np.log(dtv)/10
for nm, gv, gc in [('v.1c.1', L10, L10), ('v1c.1', np.log(dtv), L10), ('vH', np.log(dtv/3600)/10, L10), ('vM', np.log(dtv/60)/10, L10), ('v1p', np.log1p(dtv)/10, np.log1p(dtv)/10), ('cH', L10, np.log(dtv/3600)/10)]:
    print(nm, run_var(gv, gc, 3600), run_var(gv, gc, 60))
```