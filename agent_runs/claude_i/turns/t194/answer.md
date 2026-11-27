When insulin was given in the last hour, the best column-0 value is about 0.85. With no insulin ever, it is about −0.15. So column 0 behaves like a "has a previous insulin record" flag. That means the glucose register is cross-code: it carries insulin information directly into the glucose value head.

If the register also passes the last insulin rate, through its z-score and rate bin, that is a direct, editable path from dose to glucose prediction. I'll test combinations on glucose rows:
- column 0 set to the insulin flag;
- the rate bin set to the last insulin rate's bin;
- the z-score of the last insulin rate in column 4, 5, 6 or 7.

A combination with median error under 1 and bias near 0 would confirm the cross-code register.

```python
has_ins = (~np.isnan(lirG)).astype(float)
zr_ins = np.where(np.isnan(lirG), 0, zsc('insulin','rate', np.nan_to_num(lirG)))
rb_ins = np.where(np.isnan(lirG), 0, fbin('insulin','rate', np.nan_to_num(lirG, nan=0.0)))
def vm_feat(featfun, rb=None):
    N = ctxG.shape[0]; mu, ls = zprior2(ctxG)
    view = ctxG + lin2(mu,'z_head')
    q = F_.layer_norm(view + T64('type_emb.weight')[1], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in cG]); has = ~np.isnan(lvG)
    zl = np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(cG,lvG)])
    ag = np.where(has, np.log1p(np.nan_to_num(laG))/10.0, 0.0)
    feats = np.zeros((N,8)); feats[:,1] = has; feats[:,3] = ag; feats[:,2] = zl
    featfun(feats)
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(cG,lvG)])
    rbt = torch.zeros(N, dtype=torch.long) if rb is None else torch.tensor(rb, dtype=torch.long)
    wc = q + T64('embed.code.weight')[cid] + torch.tensor(feats) @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[rbt] + T64('last_status.weight')[0]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias') + T64('embed_transition.value_bin.weight')[cid*17+lb]
    p = torch.softmax(lg,-1); cen = torch.stack([CENT2[c] for c in cG])
    m = (p[:,1:]*cen).sum(1).numpy(); return np.median(np.abs(m-tG)).round(3), (m-tG).mean().round(2)
def f0(fe): fe[:,0] = has_ins
print('col0=has_ins', vm_feat(f0), '| +rb', vm_feat(f0, rb_ins))
for c_ in [4,5,6,7]:
    def fz(fe, c_=c_): fe[:,0] = has_ins; fe[:,c_] = zr_ins
    print('col0=has_ins, z_rate in', c_, vm_feat(fz), '| +rb', vm_feat(fz, rb_ins))
```