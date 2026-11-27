All 720 rows were matched to a previous state, and 93% have an earlier same-code value. Next I'll run the grid search over type index, view-gap scaling, input layout and age units, and print the best few settings by mean absolute error against the tool's means.

```python
rng = np.random.default_rng(0); S = 16
out0 = X4s[pv, :192]; h0 = X4s[pv, 192:960].reshape(-1, 4, 192)
hf = h0 * np.exp(-dtv[:, None, None] * np.exp(P0['core.flow.a']))[:, None, :] if False else h0 * np.exp(-dtv[:, None, None] * np.exp(P0['core.flow.a'])[None, None, :])
on0 = LNn(hf.mean(1), P0['core.out_norm.weight'], P0['core.out_norm.bias'])
gapf = lambda x: li(gel(li(x[:, None], 'gap.0', P0)), 'gap.2', P0)
ctx0 = out0 + on0 + gapf(np.log(dtv)/10)
zp = li(gel(li(ctx0, 'z_prior.0', P0)), 'z_prior.2', P0); zm, zls = zp[:, :32], zp[:, 32:]
zsm = zm[:, None, :] + np.exp(zls)[:, None, :] * rng.standard_normal((len(R), S, 32))
zh = li(zsm, 'z_head', P0)                       # (N,S,192)
cent = np.array([VOCAB['value_bin_centres'][cname[c]] for c in cv])
def head_mean(type_i, gscale, layout, age_unit, P=P0, extra=None):
    view = (out0 + on0 + gapf(np.log(dtv)*gscale))[:, None, :] + zh
    q = LNn(view + P['type_emb.weight'][type_i], P['query_norm.weight'], P['query_norm.bias'])
    a = np.where(has, np.log1p(np.nan_to_num(agev)/(1 if age_unit == 's' else 3600))/10, 0)
    num6 = np.stack([zl_, has*1.0, 0*zl_, 0*zl_, 0*zl_, 0*zl_], 1)
    feat = np.concatenate([has[:, None]*1.0, a[:, None], num6], 1) if layout == 0 else np.concatenate([num6, has[:, None]*1.0, a[:, None]], 1)
    wc = q + (P['embed.code.weight'][cv] + li(feat, 'last_numeric', P) + P['embed.bins.value.weight'][lvb] + P['embed.bins.rate.weight'][0] + P['last_status.weight'][0])[:, None, :]
    if extra is not None: wc = wc + extra
    lg = li(gel(li(wc, 'attr_heads.value_bin.0', P)), 'attr_heads.value_bin.2', P) + P['embed_transition.value_bin.weight'][cv*17 + lvb][:, None, :]
    p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); p = p.mean(1)
    return (p[:, 1:] * cent).sum(1), p
res = []
for t, gs, lay, au in itertools.product(range(15), [1.0, 0.1], [0, 1], ['s', 'h']):
    m_, _ = head_mean(t, gs, lay, au); res.append((np.abs(m_ - tool).mean(), t, gs, lay, au))
res.sort(); print([tuple(np.round(x, 3) if isinstance(x, float) else x for x in r) for r in res[:6]])
```