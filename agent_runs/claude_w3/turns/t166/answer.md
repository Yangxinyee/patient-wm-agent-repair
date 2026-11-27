The prior mean matches the saved z better (correlation 0.68) than my posterior does (0.61). So my posterior inputs are adding mismatched information, or the context is still off. Earlier, removing the time flow (0.41) or the gap term (0.53) each helped, which suggests the context really is mis-specified. I'll run a small grid over the time flow (on or off) and the gap term (on or off) on all 12 stays and compare relative errors. If the best cell lands well below 0.3, I've found the context convention; otherwise the encoders are the remaining suspect.

```python
def zp_grid(flow, gap, kmax=200, eu=True):
    errs = []
    for sid in np.unique(F4['stay_id']):
        m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(int(sid))
        for k in range(1, min(kmax, len(B))):
            e, u = enc2(B[k])
            if not eu: e, u = e*0, u*0
            dt = (hr_[k]-hr_[k-1])*3600.; hp = Xs[k-1,192:960].reshape(4,192)
            if flow: hp = hp*torch.exp(-dt*torch.exp(fa_))
            ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + (MLP(torch.tensor([np.log(dt)/10.]), 'gap') if gap else 0)
            zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
            errs.append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
    return round(float(np.median(errs)), 3)
print([(f, g, zp_grid(f, g)) for f in (True, False) for g in (True, False)])
```