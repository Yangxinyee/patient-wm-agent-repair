The architecture says the memory holds bundle vectors e, and that drug atoms (including infusion-end atoms) go only into u. Drug-only bundles therefore have an "empty" e. So the open questions are whether drug-only bundles enter memory and whether the current bundle is already in memory during its own step. I'll write my own one-step check with explicit memory choices: all earlier bundles, earlier non-drug bundles, and each of those plus the current bundle. I'll compare against the tool's states for bundles 4–12 of one stay. The variant with tiny errors everywhere is the true rule.

```python
sid_a=np.asarray(S['stay_id'])
def my_one(stay,k,variant):
    bl_=BUN[stay]; idx_=np.where(sid_a==stay)[0]; Es=[bvd(b[1]) for b in bl_]
    tms=np.array([b[0]*60.0 for b in bl_]); dro=[all(a[0] in ('insulin','norepinephrine') for a in b[1]) for b in bl_]
    hi_=k+1 if variant in ('C','D') else k
    js=[j for j in range(hi_) if (variant in ('A','C') or not dro[j])][-128:]
    Mem=torch.stack([Es[j][0] for j in js]); ages=torch.tensor([tms[k]-tms[j] for j in js],dtype=torch.float64)
    o_,y_=step(Xz[idx_[k-1],:192], Xz[idx_[k-1],192:960].reshape(4,192), float(tms[k]-tms[k-1]), Es[k][0], Es[k][1], Xz[idx_[k],960:992], Mem, ages)
    return round(max(float((o_-Xz[idx_[k],:192]).abs().max()), float((y_.reshape(-1)-Xz[idx_[k],192:960]).abs().max())),5)
print('n rows', (sid_a==st_).sum(), 'n bundles', len(BUN[st_]))
for vv in 'ABCD': print(vv, [my_one(st_,k_,vv) for k_ in range(4,13)])
```