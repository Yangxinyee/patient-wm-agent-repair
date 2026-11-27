All 5,673 fwd_3 readings were matched to their bundles, and the local gap MLP agrees with the earlier helper to 5e-7, so the head's input side is in place.

Now the exhaustive search for the head's two unknowns: the `type_emb` index (15 options) and the `last_status` row (2), together with where the 8 `last_numeric` inputs go. The layouts place "has last" (=1), log-age/10 and the last value's z-score in distinct columns, optionally with one extra constant flag column. I'll score each candidate by RMSE against fwd_3's predicted means on 600 random readings that have a previous value, then report the best few. One configuration far ahead of the rest, with a small RMSE, would settle the head.

```python
rng_ = np.random.default_rng(0)
idx_h = [i for i,h in enumerate(H) if h['has']]
sub_ = rng_.choice(idx_h, 600, replace=False)
Vs = torch.stack([H[i]['view'] for i in sub_]); cids = torch.tensor([H[i]['cid'] for i in sub_])
lvbs = torch.tensor([H[i]['lvb'] for i in sub_]); ages = torch.tensor([H[i]['age'] for i in sub_], dtype=torch.float64)
zvs = torch.tensor([H[i]['zv'] for i in sub_], dtype=torch.float64); tg_ = torch.tensor([H[i]['target'] for i in sub_], dtype=torch.float64)
CENr = torch.stack([CEN[H[i]['code']] for i in sub_])
W1, b1 = W['attr_heads.value_bin.0.weight'], W['attr_heads.value_bin.0.bias']
W2, b2 = W['attr_heads.value_bin.2.weight'], W['attr_heads.value_bin.2.bias']
Bm = W1 @ W['last_numeric.weight']                      # 192x8
TR = W['embed_transition.value_bin.weight'][cids*17 + lvbs]   # N x 17
cands = []
for ph, pa, pz in itertools.permutations(range(8), 3):
    for pf in [None] + [j for j in range(8) if j not in (ph,pa,pz)]:
        cands.append((ph,pa,pz,pf))
print("n layouts", len(cands))
res_s = []
t0_=time.time()
for ty in range(15):
    Q = LNt(Vs + W['type_emb.weight'][ty], 'query_norm')
    for stt in (0,1):
        base = Q + W['embed.code.weight'][cids] + W['embed.bins.value.weight'][lvbs] + W['last_numeric.bias'] + W['last_status.weight'][stt]
        A = base @ W1.T + b1
        for c0 in range(0, len(cands), 252):
            cc = cands[c0:c0+252]
            ph = torch.tensor([c[0] for c in cc]); pa = torch.tensor([c[1] for c in cc]); pz = torch.tensor([c[2] for c in cc])
            add = Bm[:,ph].T[:,None,:] + ages[None,:,None]*Bm[:,pa].T[:,None,:] + zvs[None,:,None]*Bm[:,pz].T[:,None,:]
            fl = torch.stack([Bm[:,c[3]] if c[3] is not None else torch.zeros(192,dtype=torch.float64) for c in cc])
            hid = gelu(A[None] + add + fl[:,None,:])
            lg = hid @ W2.T + b2 + TR[None]
            p = torch.softmax(lg, -1)
            mu = (p[...,1:]*CENr[None]).sum(-1)
            rm = torch.sqrt(((mu - tg_[None])**2).mean(-1))
            for j,c in enumerate(cc): res_s.append((float(rm[j]), ty, stt, c))
res_s.sort(key=lambda x: x[0])
print("time %.1fs" % (time.time()-t0_))
for r_ in res_s[:8]: print(" rmse %.3f type %d status %d layout(has,age,zv,flag) %s" % r_)
print("target sd:", float(tg_.std()))
```