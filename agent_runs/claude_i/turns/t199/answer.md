Using a shared glucose register changes nothing; the median error stays at 3.6. The remaining glucose-specific input I haven't varied is the query's type embedding. Glucose may use a different type index from MAP, which matches well at type 1.

I'll scan all 15 type indices on the glucose rows, with column 0 both off and set to the insulin flag. If one index brings the median error near 1, the head is resolved. If none does, I'll treat the glucose head as approximate and calibrate through the tool.

```python
def vm_type(typ, col0):
    N = ctxG.shape[0]; mu, ls = zprior2(ctxG)
    view = ctxG + lin2(mu,'z_head')
    q = F_.layer_norm(view + T64('type_emb.weight')[typ], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in cG]); has = ~np.isnan(lvG)
    zl = np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(cG,lvG)])
    ag = np.where(has, np.log1p(np.nan_to_num(laG))/10.0, 0.0)
    feats = np.zeros((N,8)); feats[:,1] = has; feats[:,3] = ag; feats[:,2] = zl; feats[:,0] = col0
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(cG,lvG)])
    wc = q + T64('embed.code.weight')[cid] + torch.tensor(feats) @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[0] + T64('last_status.weight')[0]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias') + T64('embed_transition.value_bin.weight')[cid*17+lb]
    p = torch.softmax(lg,-1); cen = torch.stack([CENT2[c] for c in cG])
    m = (p[:,1:]*cen).sum(1).numpy(); return np.median(np.abs(m-tG)).round(2), (m-tG).mean().round(2)
print([(t_, vm_type(t_, 0.0), vm_type(t_, has_ins)) for t_ in range(15)])
```