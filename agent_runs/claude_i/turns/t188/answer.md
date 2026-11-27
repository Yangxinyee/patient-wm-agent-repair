Sampling z did not help; the error stays around 4–5 mg/dL. So the prior-z treatment is not the problem. The feature layout inside `last_numeric`, the status index or the type index must be wrong.

On 1,500 rows, I'll compare permutations that place has-last, log-age and z-last-value in different columns, together with status 0 or 1. I'll report the error by code for the best candidates. A configuration with a median error near 1 mg/dL identifies the layout. If none gets close, the issue lies elsewhere, for example in how ctx is formed for the head.

```python
import itertools
sub_i = np.arange(0, len(Fm), 3)[:1500]
ctxS = ctxF[sub_i]; codesS = Fm.event.values[sub_i].tolist(); lvS = Fm.lastv.values[sub_i]; laS = lastage[sub_i]; tgtS = Fm.predicted_mean.values[sub_i]
def vm_custom(ctx_, codes, lastv, lastage_s, pos, status=0, typ=1, agediv=10.0):
    N = ctx_.shape[0]; mu, ls = zprior2(ctx_)
    view = ctx_ + lin2(mu,'z_head')
    q = F_.layer_norm(view + T64('type_emb.weight')[typ], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in codes]); has = ~np.isnan(lastv)
    zl = np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(codes,lastv)])
    ag = np.where(has, np.log1p(np.nan_to_num(lastage_s))/agediv, 0.0)
    feats = np.zeros((N,8)); feats[:,pos[0]] = has; feats[:,pos[1]] = ag; feats[:,pos[2]] = zl
    feats = torch.tensor(feats)
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(codes,lastv)])
    wc = q + T64('embed.code.weight')[cid] + feats @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[0] + T64('last_status.weight')[status]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias') + T64('embed_transition.value_bin.weight')[cid*17+lb]
    p = torch.softmax(lg,-1); cen = torch.stack([CENT2[c] for c in codes])
    return (p[:,1:]*cen).sum(1).numpy()
resl = []
for pos in itertools.permutations(range(4),3):
    for status in [0,1]:
        m_ = vm_custom(ctxS, codesS, lvS, laS/3600, pos, status)
        resl.append((np.median(np.abs(m_-tgtS)), pos, status))
resl.sort(key=lambda r: r[0])
print([(round(a,3),p,s) for a,p,s in resl[:6]])
best_pos, best_st = resl[0][1], resl[0][2]
m_ = vm_custom(ctxS, codesS, lvS, laS/3600, best_pos, best_st)
for ev_ in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    w_ = np.array(codesS)==ev_
    print(ev_, w_.sum(), 'MAD med', np.median(np.abs(m_-tgtS)[w_]).round(3))
```