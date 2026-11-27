The traceback is a slicing bug: I took `zp[:32]` on a batch, which slices rows instead of columns. Rather than recovering the z-free view by fixed-point iteration, I'll recompute it directly in the bundle loop (6 s) and store it per reading, along with the stay id and the observed value's fine bin. Then I'll run the same comparison: prior-mean z versus 64 draws, with the prior's second half read as log-variance or as log-std. I'll check both the means and the fine-bin NLL against fwd_3.

```python
t0_=time.time()
for h in H: h.pop('v0', None)
H2 = []
for sid in bundles_B.keys():
    R = run_tf(sid); last = {}
    for k, r in enumerate(R):
        if k > 0:
            pv = R[k-1]
            hfl = pv['y'] * torch.exp(-r['dt']*fa_)
            g_ = W['gap.2.weight'] @ gelu(W['gap.0.weight'][:,0]*(math.log(r['dt'])/10) + W['gap.0.bias']) + W['gap.2.bias']
            v0 = pv['out'] + LNt(hfl.mean(0)[None], 'core.out_norm')[0] + g_
            for a in r['ats']:
                c = a['code']
                if c in VCODES and a['kind']==0 and a.get('value') is not None and (int(sid), int(round(r['t'])), c) in tgt:
                    L = last.get(c)
                    H2.append(dict(sid=int(sid), hour=int(round(r['t'])), k=k, v0=v0, cid=CID[c], code=c, has=L is not None,
                                   lvb=fbv(c, L[0]) if L else 0, age=math.log1p((r['t']-L[1])*3600)/10 if L else 0.0,
                                   zv=zsv(c, L[0]) if L else 0.0, target=tgt[(int(sid), int(round(r['t'])), c)], obs=a['value']))
        for a in r['ats']:
            if a['code'] in VCODES and a['kind']==0 and a.get('value') is not None:
                last[a['code']] = (a['value'], r['t'])
H = H2
print("rows", len(H), "time %.1fs" % (time.time()-t0_))
cid = torch.tensor([h['cid'] for h in H]); lvb = torch.tensor([h['lvb'] for h in H])
f8 = torch.zeros(len(H), 8, dtype=torch.float64)
f8[:,0] = torch.tensor([1.0 if h['has'] else 0.0 for h in H], dtype=torch.float64)
f8[:,3] = torch.tensor([h['age'] for h in H], dtype=torch.float64); f8[:,6] = torch.tensor([h['zv'] for h in H], dtype=torch.float64)
cen = torch.stack([CEN[h['code']] for h in H])
fixed = W['embed.code.weight'][cid] + W['embed.bins.value.weight'][lvb] + f8 @ W['last_numeric.weight'].T + W['last_numeric.bias'] + W['last_status.weight'][0]
TRr = W['embed_transition.value_bin.weight'][cid*17 + lvb]
V0 = torch.stack([h['v0'] for h in H]); tg_all = torch.tensor([h['target'] for h in H], dtype=torch.float64)
obs_bin = torch.tensor([fbv(h['code'], h['obs']) for h in H])
f3i = f3.copy(); f3i['hour'] = f3i['hour'].round().astype(int); f3i = f3i.set_index(['stay_id','hour','event'])
nll3 = torch.tensor(f3i.loc[[(h['sid'],h['hour'],h['code']) for h in H], 'nll'].values, dtype=torch.float64)
def probs_from_v0(V0_, fixed_=None, TR_=None, nz=0, lv='logvar', seed=0):
    fixed_ = fixed if fixed_ is None else fixed_; TR_ = TRr if TR_ is None else TR_
    zp = lin_t(gelu(lin_t(V0_,'z_prior.0')),'z_prior.2'); mu_, s2 = zp[:, :32], zp[:, 32:]
    g = torch.Generator().manual_seed(seed)
    sd_ = torch.exp(0.5*s2) if lv=='logvar' else torch.exp(s2)
    zs = [mu_] if nz == 0 else [mu_ + sd_*torch.randn(mu_.shape, generator=g, dtype=torch.float64) for _ in range(nz)]
    P_ = 0
    for z_ in zs:
        q_ = LNt(V0_ + lin_t(z_, 'z_head') + W['type_emb.weight'][1], 'query_norm')
        P_ = P_ + torch.softmax(gelu((q_ + fixed_) @ W1.T + b1) @ W2.T + b2 + TR_, -1)
    return P_/len(zs)
for mode in [('mean',0,'logvar'), ('s64_logvar',64,'logvar'), ('s64_logstd',64,'logstd')]:
    P_ = probs_from_v0(V0, nz=mode[1], lv=mode[2])
    mu = (P_[:,1:]*cen).sum(-1); nl = -torch.log(P_[torch.arange(len(H)), obs_bin])
    e2 = (mu-tg_all)**2
    print(mode[0], "rmse all %.3f gf %.3f mapart %.3f | mean NLL mine %.4f fwd3 %.4f | rmse nll %.4f" % (
        float(e2.mean().sqrt()), float(e2[cid==6].mean().sqrt()), float(e2[cid==7].mean().sqrt()),
        float(nl.mean()), float(nll3.mean()), float(((nl-nll3)**2).mean().sqrt())))
```