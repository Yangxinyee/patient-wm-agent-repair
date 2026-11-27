The +8 bias on finger glucose survives z sampling, and MAP still matches. The missing piece must be glucose-specific. The unused `last_numeric` columns (0, 4, 5, 6, 7) may carry a flag or value that the glucose register sets, for instance the last insulin rate.

I'll test on finger and lab rows:
- each unused column set to 1;
- the z-score of the last insulin rate placed in each of those columns;
- the rate-bin embedding indexed by the last insulin rate.

A variant that brings finger and lab bias near 0 and the median error to about 1 identifies the true register. If none does, I'll test a shared glucose register.

```python
gfw = np.isin(np.array(codesS), ['glucose_finger','glucose_lab'])
ctxG = ctxS[gfw]; cG = list(np.array(codesS)[gfw]); lvG = lvS[gfw]; laG = laS[gfw]/3600; tG = tgtS[gfw]
lirG = FS.lir.values[gfw]; liaG = FS.lia.values[gfw]
def vm_var(extra_col=None, extra_val=None, rate_bin_idx=None):
    N = ctxG.shape[0]; mu, ls = zprior2(ctxG)
    view = ctxG + lin2(mu,'z_head')
    q = F_.layer_norm(view + T64('type_emb.weight')[1], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in cG]); has = ~np.isnan(lvG)
    zl = np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(cG,lvG)])
    ag = np.where(has, np.log1p(np.nan_to_num(laG))/10.0, 0.0)
    feats = np.zeros((N,8)); feats[:,1] = has; feats[:,3] = ag; feats[:,2] = zl
    if extra_col is not None: feats[:,extra_col] = extra_val
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(cG,lvG)])
    rb = torch.zeros(N, dtype=torch.long) if rate_bin_idx is None else torch.tensor(rate_bin_idx)
    wc = q + T64('embed.code.weight')[cid] + torch.tensor(feats) @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[rb] + T64('last_status.weight')[0]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias') + T64('embed_transition.value_bin.weight')[cid*17+lb]
    p = torch.softmax(lg,-1); cen = torch.stack([CENT2[c] for c in cG])
    m = (p[:,1:]*cen).sum(1).numpy(); return np.median(np.abs(m-tG)).round(3), (m-tG).mean().round(2)
print('base', vm_var())
for c_ in [0,4,5,6,7]:
    print('col', c_, '=1:', vm_var(c_, 1.0), '| =zrate:', vm_var(c_, np.where(np.isnan(lirG),0,zsc('insulin','rate',np.nan_to_num(lirG)))))
rbi = np.where(np.isnan(lirG), 0, fbin('insulin','rate',np.nan_to_num(lirG,nan=0.0)))
print('ratebin=last insulin', vm_var(None,None,rbi))
```