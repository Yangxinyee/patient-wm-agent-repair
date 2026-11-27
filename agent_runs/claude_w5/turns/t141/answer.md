All 5,673 fwd_3 readings line up with a bundle, a tool state row and a register entry. The helpers behave as expected (z-score 0.23 and fine bin 10 for a finger reading of 176).

Now the head scan. For each of the 15 type rows I compute the local predicted mean from the tool's states (z at the prior mean, register layout as stated) and compare with the tool's predicted mean, by mean absolute error per code. The correct type should give errors far below the spread of the predictions. Because the tool averages 4 random z draws, I expect a small residual mismatch, not exact zero.

```python
X64 = torch.as_tensor(np.asarray(X_), dtype=torch.float64)
rr_i = torch.as_tensor(R_.row.values)
OUT = X64[rr_i,:192]; HH = X64[rr_i,192:960].reshape(-1,4,192)
DT = torch.as_tensor(R_.dt.values, dtype=torch.float64)
HF = HH * torch.exp(-DT[:,None]*torch.exp(T('core.flow.a')))[:,None,:]
def gapv(x): return gelu(x[:,None] @ T('gap.0.weight').T + T('gap.0.bias')) @ T('gap.2.weight').T + T('gap.2.bias')
CODES = R_.code.values
cid = torch.as_tensor([code_id[c] for c in CODES])
hl = torch.as_tensor(R_.hl.values, dtype=torch.float64)
zv = torch.as_tensor([zsc(c,'value',v) if h else 0.0 for c,v,h in zip(CODES,R_.lv.values,R_.hl.values)], dtype=torch.float64)
lb = torch.as_tensor([fbin(c,'value',v) if h else 0 for c,v,h in zip(CODES,R_.lv.values,R_.hl.values)])
lage = torch.log1p(torch.as_tensor(R_.age.values, dtype=torch.float64))/10
CEN = {c: torch.as_tensor(np.asarray(centres_of(c), dtype=float)) for c in set(CODES)}
cen_mat = torch.stack([CEN[c] for c in CODES])
def head_mean(type_i, layout, st=0, gscale=10.0, zmode='mean', nsamp=1, seed=0):
    gp = gapv(torch.log(DT)/gscale)
    ctx_ = OUT + lnorm(HF.mean(1), 'core.out_norm') + gp
    pr_ = mlp2(ctx_, 'z_prior'); mu, ls = pr_[:,:32], pr_[:,32:]
    inp = torch.zeros(len(DT), 8, dtype=torch.float64)
    for j,val in zip(layout, [hl, lage, zv, hl]): inp[:, j] = val
    base = T('embed.code.weight')[cid] + inp @ T('last_numeric.weight').T + T('last_numeric.bias') \
           + T('embed.bins.value.weight')[lb] + T('embed.bins.rate.weight')[0] + T('last_status.weight')[st]
    tr_ = T('embed_transition.value_bin.weight')[cid*17 + lb]
    g = torch.Generator().manual_seed(seed); P = 0
    for s_ in range(nsamp):
        zz_ = mu if zmode=='mean' else mu + torch.exp(ls)*torch.randn(mu.shape, generator=g, dtype=torch.float64)
        view_ = ctx_ + zz_ @ T('z_head.weight').T + T('z_head.bias')
        q_ = lnorm(view_ + T('type_emb.weight')[type_i], 'query_norm')
        P = P + torch.softmax(mlp2(q_ + base, 'attr_heads.value_bin') + tr_, -1)/nsamp
    return (P[:,1:]*cen_mat).sum(1), P
pm_t = torch.as_tensor(R_.pm.values)
res_ty=[]
for ty in range(15):
    m_, _ = head_mean(ty, [0,1,2,3])
    err = (m_-pm_t).abs()
    res_ty.append((ty, *[round(float(err[torch.as_tensor(CODES==c)].mean()),2) for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']]))
print(res_ty)
print('pm spread sd gf', float(pm_t[torch.as_tensor(CODES=='glucose_finger')].std()))
```