The `flow.a` range confirms dt in seconds (timescales from 10 minutes to about 2 years). No activation choice fixes the error. Removing the gap term slightly reduces it, which is suspicious, and removing the slot term makes it much worse. So the context is roughly right but something adds a large, consistent error.

Next I'll test whether my encoder outputs carry useful signal. I'll compare the error with e and u set to zero, and also try dt measured from the gap to the next bundle instead of the previous one. If zeroing e and u barely changes the error, my encodings are effectively wrong. If the next-bundle dt fits, the time convention is shifted.

```python
def zp_w(sid, eu='own', dtmode='prev', kmax=40):
    m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(sid); errs = []
    for k in range(1, min(kmax, len(B)-1)):
        e, u = [torch.tensor(np.asarray(t), dtype=torch.float64) for t in enc_bundle(B[k])[:2]]
        if eu == 'zero': e, u = e*0, u*0
        if eu == 'swap': e, u = u, e
        dt = ((hr_[k]-hr_[k-1]) if dtmode=='prev' else (hr_[k+1]-hr_[k]))*3600.
        hp = Xs[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa_))
        ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
        zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
        errs.append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
    return round(float(np.median(errs)), 3)
for kw in [dict(eu='zero'), dict(eu='swap'), dict(dtmode='next')]: print(kw, zp_w(s0_, **kw))
```