z_post has 50 completely dead hidden units, with activation spread under 0.01 and mean 0. The median unit's importance is only 0.01, against a total z spread of 1.42. So I can repurpose dead units to compute a gated dose deviation without changing any factual z.

That makes z_post the injection site. Designing the injection direction still needs a head I trust for glucose. The remaining finger-glucose error points to a glucose-specific register detail. One hypothesis: finger and lab share a single glucose register (the last value of either code, with its age), alongside the insulin flag in column 0. I'll test that on the glucose rows. A median error near 1 would confirm it.

```python
gl_tr = train[train.event.isin(['glucose_finger','glucose_lab']) & train.stay_id.isin(set(S3.tolist()))][['stay_id','hour','event','value']].sort_values(['stay_id','hour'])
FSg = FS[gfw].copy()
def last_any_glu(s, h):
    d = gl_tr[(gl_tr.stay_id==s)&(gl_tr.hour<h-1e-6)]
    return (d.value.iloc[-1], h-d.hour.iloc[-1]) if len(d) else (np.nan, np.nan)
la_ = np.array([last_any_glu(s,h) for s,h in zip(FSg.stay_id, FSg.hour)])
def vm_reg(lv_, la_h, col0):
    N = ctxG.shape[0]; mu, ls = zprior2(ctxG)
    view = ctxG + lin2(mu,'z_head')
    q = F_.layer_norm(view + T64('type_emb.weight')[1], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in cG]); has = ~np.isnan(lv_)
    zl = np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(cG,lv_)])
    ag = np.where(has, np.log1p(np.nan_to_num(la_h))/10.0, 0.0)
    feats = np.zeros((N,8)); feats[:,1] = has; feats[:,3] = ag; feats[:,2] = zl; feats[:,0] = col0
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(cG,lv_)])
    wc = q + T64('embed.code.weight')[cid] + torch.tensor(feats) @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[0] + T64('last_status.weight')[0]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias') + T64('embed_transition.value_bin.weight')[cid*17+lb]
    p = torch.softmax(lg,-1); cen = torch.stack([CENT2[c] for c in cG])
    m = (p[:,1:]*cen).sum(1).numpy(); return np.median(np.abs(m-tG)).round(3), (m-tG).mean().round(2)
for nm_, lv_, lah in [('same-code', lvG, laG), ('any-glucose', la_[:,0], la_[:,1])]:
    print(nm_, 'col0=0', vm_reg(lv_, lah, 0.0), 'col0=has_ins', vm_reg(lv_, lah, has_ins), 'col0=1', vm_reg(lv_, lah, 1.0))
```