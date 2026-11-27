All 5306 readings are mapped. The gap to the previous bundle is usually 3300 s: the previous bundle is the :05 drug bundle, which holds infusion starts and ends.

Next, I will evaluate the replicated value head with z at the prior mean. I'll scan the type index (0–14) and the gap scaling (log dt or log dt/10), using the documented register order [has_last, log(1+age)/10, z_last_value, z_last_rate]. The last-rate bin is 0 and last_status is row 0. Variants are ranked by correlation, regression slope and mean absolute difference against the tool's predicted means. If no variant reaches about 0.99, I'll vary the register columns and status next.

```python
P = dict(ow=Tn('core.out_norm.weight'), ob=Tn('core.out_norm.bias'), qw=Tn('query_norm.weight'), qb=Tn('query_norm.bias'),
         te=Tn('type_emb.weight'), ce=Tn('embed.code.weight'), LW=Tn('last_numeric.weight'), Lb=Tn('last_numeric.bias'),
         bv=Tn('embed.bins.value.weight'), br=Tn('embed.bins.rate.weight'), ls=Tn('last_status.weight'),
         tv=Tn('embed_transition.value_bin.weight'), zhW=Tn('z_head.weight'), zhb=Tn('z_head.bias'), fa=Tn('core.flow.a'))
def head_inputs(F, X, gap_scale=10.0):
    out = X[F.pk.values, :192]; Hs = X[F.pk.values, 192:960].reshape(-1,4,192)
    dt = F.dt.values
    hf = Hs * np.exp(-dt[:,None]*np.exp(P['fa'])[None,:])[:,None,:]
    m = lnorm(hf.mean(1), P['ow'], P['ob'])
    ctx = out + m + mlp2(np.log(dt)[:,None]/10.0, 'gap')
    g_view = mlp2(np.log(dt)[:,None]/gap_scale, 'gap')
    zp = mlp2(ctx, 'z_prior')
    return out, m, g_view, zp
def head_value(F, out, m, g_view, zp, typ, regorder=(0,1,2,3), status=0, zsamp=None):
    codes = np.array([CID[e] for e in F.event]); evs = F.event.values
    z = zp[:, :32] if zsamp is None else zsamp
    view = out + m + g_view + z @ P['zhW'].T + P['zhb']
    q = lnorm(view + P['te'][typ], P['qw'], P['qb'])
    hasl = (~np.isnan(F.lastv.values)).astype(float)
    age = np.where(hasl>0, (F.hour.values - F.lasth.values)*3600, 0.0)
    zl = np.zeros(len(F)); lb = np.zeros(len(F), int)
    for ev in np.unique(evs):
        ix = (evs==ev)&(hasl>0)
        zl[ix] = zsc(ev,'value',F.lastv.values[ix]); lb[ix] = fbin(ev,'value',F.lastv.values[ix])
    feats = np.zeros((len(F),8)); feats[:,regorder[0]] = hasl; feats[:,regorder[1]] = np.log1p(age)/10; feats[:,regorder[2]] = zl
    wc = q + P['ce'][codes] + feats @ P['LW'].T + P['Lb'] + P['bv'][lb] + P['br'][0] + P['ls'][status]
    lg = mlp2(wc, 'attr_heads.value_bin') + P['tv'][codes*17 + lb]
    lg = lg - lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    cen = np.stack([CEN[e] for e in evs]); mean = (p[:,1:]*cen).sum(1)
    return mean, p
res_=[]
for gs_ in [10.0, 1.0]:
    out_, m_, gv_, zp_ = head_inputs(F, X3, gs_)
    for typ in range(15):
        mn,_ = head_value(F, out_, m_, gv_, zp_, typ)
        y = F.predicted_mean.values
        res_.append((gs_, typ, np.corrcoef(mn,y)[0,1], np.polyfit(mn,y,1)[0], np.abs(mn-y).mean()))
res_ = sorted(res_, key=lambda r:-r[2])
for r in res_[:6]: print(np.round(r,4))
```