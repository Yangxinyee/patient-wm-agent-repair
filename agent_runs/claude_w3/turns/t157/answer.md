The old checker used a different state file, so its 0.10 isn't comparable with my test. One basic convention is still untested: how z_post's 64 outputs split into mean and log-sd. They might be interleaved, or the mean might be the second half. I'll test four slicings. If one gives a near-zero error, the earlier mismatch was just my slicing.

```python
def zp_slice(sid, sl, kmax=40):
    m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(sid); errs = []
    for k in range(1, min(kmax, len(B))):
        e, u = [torch.tensor(np.asarray(t), dtype=torch.float64) for t in enc_bundle(B[k])[:2]]
        dt = (hr_[k]-hr_[k-1])*3600.; hp = Xs[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa_))
        ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
        zm = MLP(torch.cat([ctx, e, u]), 'z_post')[sl]
        errs.append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
    return round(float(np.median(errs)), 3)
for sl in [slice(0,32), slice(32,64), slice(0,64,2), slice(1,64,2)]: print(sl, zp_slice(s0_, sl))
```