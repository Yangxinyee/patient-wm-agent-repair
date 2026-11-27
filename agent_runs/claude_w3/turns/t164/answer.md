The disassembly shows the old `zpost_err` used the same formula as my fresh code, but reported an absolute RMS error. Its 0.10 is per element, about 40% of the z scale, so the old code didn't match either.

One untested detail is the time unit inside the gap embedding: log of dt in seconds, minutes or hours, or log(1+dt), before the division by 10. I'll vary only that, keeping the time flow in seconds. If one variant drops the error sharply, that is the missing convention. If none does, the mismatch is in how the stored state relates to the next step's context.

```python
def zp_gap(sid, gfun, kmax=60):
    m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(sid); errs = []
    for k in range(1, min(kmax, len(B))):
        e, u = enc2(B[k])
        dt = (hr_[k]-hr_[k-1])*3600.; hp = Xs[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa_))
        ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([gfun(dt)]), 'gap')
        zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
        errs.append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
    return round(float(np.median(errs)), 3)
for nm_g, gf_ in [('min', lambda d: np.log(d/60)/10), ('hour', lambda d: np.log(d/3600)/10), ('log1p', lambda d: np.log1p(d)/10), ('day', lambda d: np.log(d/86400)/10), ('sec_raw/100', lambda d: np.log(d)/100)]:
    print(nm_g, zp_gap(s0_, gf_))
```