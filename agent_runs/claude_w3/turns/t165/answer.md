Measuring the gap in seconds is confirmed as best, so that convention is right. Next I'll pool all 12 stays and compare my reconstructed z with the saved z, per dimension and in overall scale. I'll do the same for the prior mean `z_prior(ctx)`. High correlation with a wrong scale would point to a missing transform. If the prior mean correlates better than my posterior, my e/u inputs are mostly adding noise.

```python
def zp_all(kmax=200):
    P, Q, Zs = [], [], []
    for sid in np.unique(F4['stay_id']):
        m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(int(sid))
        for k in range(1, min(kmax, len(B))):
            e, u = enc2(B[k])
            dt = (hr_[k]-hr_[k-1])*3600.; hp = Xs[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa_))
            ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
            P.append(MLP(torch.cat([ctx, e, u]), 'z_post')[:32]); Q.append(MLP(ctx, 'z_prior')[:32]); Zs.append(Xs[k,960:])
    return torch.stack(P).numpy(), torch.stack(Q).numpy(), torch.stack(Zs).numpy()
Pz, Qz, Zz = zp_all()
cor = lambda a, b: np.array([np.corrcoef(a[:,i], b[:,i])[0,1] for i in range(32)])
print('n', len(Zz), 'post corr med', np.round(np.median(cor(Pz, Zz)),3), 'prior corr med', np.round(np.median(cor(Qz, Zz)),3))
print('slope post', np.round(np.median([np.polyfit(Pz[:,i], Zz[:,i], 1)[0] for i in range(32)]),3), 'sd ratio', np.round(np.median(Zz.std(0)/Pz.std(0)),3))
```