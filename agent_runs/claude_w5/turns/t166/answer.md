The search singles out one configuration clearly: type index 1, status row 0, and `last_numeric` inputs in columns 0 (has-last), 3 (log-age/10) and 6 (last value z-score). RMSE is 1.25 mg/dL against a target SD of 57; the next-best layout is at 1.9. This matches the defaults left in my old helper, so the earlier runs had probably found the same thing.

The leftover 1.25 is most likely z noise: fwd_3 averages 4 random prior draws, while I used the prior mean. To check, I'll evaluate all 5,673 readings with the prior mean and with a 64-draw average. I'll try both readings of the prior's second half (log-variance and log-std); whichever lowers the RMSE is the right one. I'll also compare my per-reading fine-bin NLL with fwd_3's `nll` column. A close match would make the local head a full substitute for factual forward passes when screening edits.

```python
def head_eval(rows, ty=1, stt=0, zmode='mean', nz=64, seed=0, lv='logvar', extra_view=None):
    Vv = torch.stack([h['view'] for h in rows])
    cid = torch.tensor([h['cid'] for h in rows]); lvb = torch.tensor([h['lvb'] for h in rows])
    f8 = torch.zeros(len(rows), 8, dtype=torch.float64)
    f8[:,0] = torch.tensor([1.0 if h['has'] else 0.0 for h in rows], dtype=torch.float64)
    f8[:,3] = torch.tensor([h['age'] for h in rows], dtype=torch.float64)
    f8[:,6] = torch.tensor([h['zv'] for h in rows], dtype=torch.float64)
    cen = torch.stack([CEN[h['code']] for h in rows])
    fixed = W['embed.code.weight'][cid] + W['embed.bins.value.weight'][lvb] + f8 @ W['last_numeric.weight'].T + W['last_numeric.bias'] + W['last_status.weight'][stt]
    TRr = W['embed_transition.value_bin.weight'][cid*17 + lvb]
    # recover v0 (view without z) from stored view: stored view used prior-mean z
    v0 = torch.stack([h['v0'] for h in rows]) if 'v0' in rows[0] else None
    return Vv, fixed, TRr, cen, cid, lvb
# need v0 per row: recompute by subtracting z_head(prior mean) is ambiguous -> store v0 now
for h in H:
    pass
def v0_of(view):
    # fixed-point: view = v0 + z_head(mu(v0)); solve by iteration
    v = view.clone()
    for _ in range(30):
        zp = lin_t(gelu(lin_t(v, 'z_prior.0')), 'z_prior.2')
        v = view - lin_t(zp[:32], 'z_head')
    return v
Vall = torch.stack([h['view'] for h in H])
V0 = v0_of(Vall)
chk = (V0 + lin_t(lin_t(gelu(lin_t(V0,'z_prior.0')),'z_prior.2')[:, :32], 'z_head') - Vall).abs().max()
print("v0 recovery check:", float(chk))
Vv, fixed, TRr, cen, cid, lvb = head_eval(H)
tg_all = torch.tensor([h['target'] for h in H], dtype=torch.float64)
def probs_from_v0(V0_, nz=0, lv='logvar', seed=0):
    zp = lin_t(gelu(lin_t(V0_,'z_prior.0')),'z_prior.2'); mu_, s2 = zp[:, :32], zp[:, 32:]
    if nz == 0: zs = [mu_]
    else:
        g = torch.Generator().manual_seed(seed)
        sd_ = torch.exp(0.5*s2) if lv=='logvar' else torch.exp(s2)
        zs = [mu_ + sd_*torch.randn(mu_.shape, generator=g, dtype=torch.float64) for _ in range(nz)]
    P_ = 0
    for z_ in zs:
        q_ = LNt(V0_ + lin_t(z_, 'z_head') + W['type_emb.weight'][1], 'query_norm')
        lg = gelu((q_ + fixed) @ W1.T + b1) @ W2.T + b2 + TRr
        P_ = P_ + torch.softmax(lg, -1)
    return P_/len(zs)
obs_bin = torch.tensor([fbv(h['code'], h['obs']) for h in H])
nll3 = torch.tensor(f3.set_index(['stay_id','hour','event']).loc[[(h_k[0],h_k[1],h_k[2]) for h_k in [(int(k[0]),int(k[1]),k[2]) for k in [(0,0,0)]]]].shape[0] if False else 0)
f3i = f3.copy(); f3i['hour'] = f3i['hour'].round().astype(int); f3i = f3i.set_index(['stay_id','hour','event'])
keys_ = [(int(h['sid']) if 'sid' in h else None) for h in H]
for mode in [('mean',0,'logvar'), ('s64_logvar',64,'logvar'), ('s64_logstd',64,'logstd')]:
    P_ = probs_from_v0(V0, nz=mode[1], lv=mode[2])
    mu = (P_[:,1:]*cen).sum(-1)
    print(mode[0], "rmse vs fwd_3 mean: all %.3f" % float(torch.sqrt(((mu-tg_all)**2).mean())),
          "| gf %.3f" % float(torch.sqrt(((mu-tg_all)**2)[cid==6].mean())), "| map_art %.3f" % float(torch.sqrt(((mu-tg_all)**2)[cid==7].mean())))
```