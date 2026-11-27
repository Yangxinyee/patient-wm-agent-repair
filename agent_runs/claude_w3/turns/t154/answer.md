Zeroing e and u makes the error worse (1.04), so my encodings carry real signal. Next I'll test a different row convention: maybe each saved row holds the state before its bundle (out and h before step k) together with that bundle's z. I'll build the context from the same row instead of the previous one. A large error drop would confirm the row convention; no change rules it out.

```python
def zp_same(sid, kmax=40, flow=True):
    m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(sid); errs = []
    for k in range(1, min(kmax, len(B))):
        e, u = [torch.tensor(np.asarray(t), dtype=torch.float64) for t in enc_bundle(B[k])[:2]]
        dt = (hr_[k]-hr_[k-1])*3600.; hp = Xs[k,192:960].reshape(4,192)
        if flow: hp = hp*torch.exp(-dt*torch.exp(fa_))
        ctx = Xs[k,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
        zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
        errs.append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
    return round(float(np.median(errs)), 3)
print(zp_same(s0_), zp_same(s0_, flow=False))
```