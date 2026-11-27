The slot-major layout is confirmed: out equals out_norm(mean of the slots) exactly. Note that h is very large (about 64 per entry, root-mean-square). The z means have norm about 1.4, so a 0.4 relative error is real.

Next I'll split the z error by bundle type: measurement-only, action-only, and both. For action-only bundles I'll also try e = 0 and u = 0 conventions. If the error sits only in action bundles, the action encoding or the empty-set convention is the culprit. If it is everywhere, the context term is.

```python
def zp_detail(sid, flow=True, e_empty='enc'):
    m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; hi_ = F4['has_insulin'][m] | F4['has_norepinephrine'][m]
    B = stay_bundles(sid); out = []
    for k in range(1, len(B)):
        e, u = [torch.tensor(np.asarray(t), dtype=torch.float64) for t in enc_bundle(B[k])[:2]]
        nmeas = int(F4['n_atoms'][m][k]) - int(hi_[k])
        dt = (hr_[k]-hr_[k-1])*3600.
        hp = Xs[k-1,192:960].reshape(4,192)
        if flow: hp = hp*torch.exp(-dt*torch.exp(Wd('core.flow.a')))
        if e_empty == 'zero' and hi_[k] and u.abs().sum() > 0 and e.abs().sum() > 0 and nmeas <= 0: e = e*0
        ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
        zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
        out.append((int(hi_[k]), ((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item(), float(u.norm()), float(e.norm())))
    return np.array(out)
dd_ = zp_detail(s0_)
for a in (0,1): print('act', a, (dd_[:,0]==a).sum(), np.round(np.median(dd_[dd_[:,0]==a,1]),3), 'u/e norms', np.round(np.median(dd_[dd_[:,0]==a,2:],0),2))
```