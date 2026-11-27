With the empty-set conventions on both encoders and the time flow on, measurement bundles now match closely: median error 0.057, though the 90th percentile is 0.41. The overall 90th percentile is still 1.2, so some bundle types are badly off. I'll split the error by bundle composition: drug codes present, whether there are start atoms (kind 0) or end atoms (kind 2), and whether measurements are mixed in. The type with a high error then shows which encoding detail is still wrong.

```python
def comp(g):
    cs = g.code.astype(int).values; ks = g.kind.astype(int).values
    a = [f"{'A' if c==8 else 'B'}{k}" for c, k in zip(cs, ks) if c in (8, 9)]
    nm = sum(1 for c in cs if c not in (8, 9))
    return ('M' if nm else '') + '+'.join(sorted(set(a)))
res_c = defaultdict(list)
for sid in np.unique(F4['stay_id']):
    m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(int(sid))
    for k in range(1, len(B)):
        e, u = enc3(B[k]); dt = (hr_[k]-hr_[k-1])*3600.
        hp = Xs[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa_))
        ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
        zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
        res_c[comp(B[k])].append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
print({k: (len(v), round(float(np.median(v)),3), round(float(np.percentile(v,90)),3)) for k, v in res_c.items()})
```