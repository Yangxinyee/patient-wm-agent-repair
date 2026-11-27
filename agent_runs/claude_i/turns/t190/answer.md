My glucose predictions sit systematically above the tool's: +8.5 for finger and +4.2 for lab, even with no insulin. So this is a code-specific offset in my head, not a dose effect.

I'll check three things:
- whether the bin and status tables are code-indexed, from their shapes;
- whether averaging probabilities over sampled z, which glucose may be sensitive to, removes the offset;
- whether a glucose register shared between finger and lab readings does.

The variant that brings the glucose bias near 0, with the median error near 1, is the correct head.

```python
for n_ in ['embed.bins.value.weight','embed.bins.rate.weight','last_status.weight','type_emb.weight','embed_transition.value_bin.weight','last_numeric.weight']:
    print(n_, tuple(SHAPES[n_]))
gw = np.array(codesS)!='x'
def vm_nz(ctx_, codes, lastv, lastage_h, pos, status=0, typ=1, nz=32, seed=0):
    N = ctx_.shape[0]; mu, ls = zprior2(ctx_)
    g = torch.Generator().manual_seed(seed); eps = torch.randn((nz,N,32), generator=g, dtype=torch.float64)
    Z = mu[None] + torch.exp(ls)[None]*eps
    view = ctx_[None] + lin2(Z,'z_head')
    q = F_.layer_norm(view + T64('type_emb.weight')[typ], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in codes]); has = ~np.isnan(lastv)
    zl = np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(codes,lastv)])
    ag = np.where(has, np.log1p(np.nan_to_num(lastage_h))/10.0, 0.0)
    feats = np.zeros((N,8)); feats[:,pos[0]] = has; feats[:,pos[1]] = ag; feats[:,pos[2]] = zl
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(codes,lastv)])
    wc = q + (T64('embed.code.weight')[cid] + torch.tensor(feats) @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[0] + T64('last_status.weight')[status])[None]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias') + T64('embed_transition.value_bin.weight')[cid*17+lb][None]
    p = torch.softmax(lg,-1).mean(0); cen = torch.stack([CENT2[c] for c in codes])
    return (p[:,1:]*cen).sum(1).numpy()
m32 = vm_nz(ctxS, codesS, lvS, laS/3600, best_pos, best_st)
for ev_ in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    w_ = np.array(codesS)==ev_
    print(ev_, 'nz32 MAD med', np.median(np.abs(m32-tgtS)[w_]).round(3), 'bias', (m32-tgtS)[w_].mean().round(2))
```