Including the current bundle in memory (variant C) is clearly right: bundle 4 matches to 7e-5. Errors remain at alternating bundles. Those are likely the drug-only bundles (h+0.083), where my e vector is probably wrong, both as the step input and as the memory entry. I'll test, for drug-only bundles, input e ∈ {encoder output, zero} and memory entry ∈ {encoder e, zero, u, skipped}, keeping the current bundle in memory. The combination giving about 1e-4 everywhere fixes the replica.

```python
def my_one2(stay,k,tok,mem):
    bl_=BUN[stay]; idx_=np.where(sid_a==stay)[0]; Es=[bvd(b[1]) for b in bl_]
    tms=np.array([b[0]*60.0 for b in bl_]); dro=[all(a[0] in ('insulin','norepinephrine') for a in b[1]) for b in bl_]
    def ev(j): return torch.zeros(192,dtype=torch.float64) if (dro[j] and tok=='zero') else Es[j][0]
    def mv_(j):
        if not dro[j]: return Es[j][0]
        return {'e':Es[j][0],'zero':torch.zeros(192,dtype=torch.float64),'u':Es[j][1]}.get(mem)
    js=[j for j in range(k+1) if not (dro[j] and mem=='skip')][-128:]
    Mem=torch.stack([mv_(j) for j in js]); ages=torch.tensor([tms[k]-tms[j] for j in js],dtype=torch.float64)
    o_,y_=step(Xz[idx_[k-1],:192], Xz[idx_[k-1],192:960].reshape(4,192), float(tms[k]-tms[k-1]), ev(k), Es[k][1], Xz[idx_[k],960:992], Mem, ages)
    return round(max(float((o_-Xz[idx_[k],:192]).abs().max()), float((y_.reshape(-1)-Xz[idx_[k],192:960]).abs().max())),4)
dro0=[all(a[0] in ('insulin','norepinephrine') for a in b[1]) for b in BUN[st_]]; print('drug-only k in 4..12:', [k_ for k_ in range(4,13) if dro0[k_]])
for tok in ['e','zero']:
    for mem in ['e','zero','u','skip']: print(tok, mem, [my_one2(st_,k_,tok,mem) for k_ in range(4,13)])
```