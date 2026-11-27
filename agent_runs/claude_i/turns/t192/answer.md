Setting `last_numeric` column 0 to 1 for glucose rows removes most of the bias: from +7.6 to −0.7, with the median error falling from 6.3 to 4.0. So column 0 carries something glucose-specific, and possibly continuous.

To find out what it encodes, I'll search, per glucose row, for the column-0 value that best reproduces the tool's mean. I'll then correlate that best value with candidates: time since last insulin, the last insulin rate, whether insulin is running, and the age of the last value. A tight relation with one of these fixes the head. A constant best value of about 1 would mean it is a simple flag and the remaining error lies elsewhere.

```python
grid_ = np.linspace(-1, 3, 21)
def vm_col0(vals):
    N = ctxG.shape[0]; mu, ls = zprior2(ctxG)
    view = ctxG + lin2(mu,'z_head')
    q = F_.layer_norm(view + T64('type_emb.weight')[1], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in cG]); has = ~np.isnan(lvG)
    zl = np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(cG,lvG)])
    ag = np.where(has, np.log1p(np.nan_to_num(laG))/10.0, 0.0)
    feats = np.zeros((N,8)); feats[:,1] = has; feats[:,3] = ag; feats[:,2] = zl; feats[:,0] = vals
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(cG,lvG)])
    wc = q + T64('embed.code.weight')[cid] + torch.tensor(feats) @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[0] + T64('last_status.weight')[0]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias') + T64('embed_transition.value_bin.weight')[cid*17+lb]
    p = torch.softmax(lg,-1); cen = torch.stack([CENT2[c] for c in cG])
    return (p[:,1:]*cen).sum(1).numpy()
Mg = np.stack([vm_col0(np.full(len(tG), v)) for v in grid_])
bestv = grid_[np.abs(Mg - tG[None]).argmin(0)]
besterr = np.abs(Mg - tG[None]).min(0)
print('best col0 value dist', np.percentile(bestv,[10,25,50,75,90]).round(2), 'err at best med', np.median(besterr).round(2))
cand = {'has':(~np.isnan(lvG)).astype(float), 'age_h':np.nan_to_num(laG), 'lir':np.nan_to_num(lirG), 'lia':np.nan_to_num(liaG,nan=99), 'is_gf':(np.array(cG)=='glucose_finger').astype(float), 'zl':np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(cG,lvG)])}
for k_,v_ in cand.items(): print(k_, np.corrcoef(v_, bestv)[0,1].round(3), end=' | ')
```