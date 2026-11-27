The activation search did not help either. The best error is still 0.31, and the "≤" bin rule changes nothing. So the gap between my z-posterior and the model's lies elsewhere.

Next I will look at the shape of the error rather than its size. I will compare my z-posterior means with the true ones overall: magnitudes, correlation, and the regression slope. I will also test whether the truth is better explained by my prior mean, or by prior plus a posterior correction. A high correlation with a slope well below 1 would point to a squashing or scaling step. A low correlation would mean the context or encoder input is wrong.

```python
def zpair(stays_):
    mine, prior_, true_ = [], [], []
    for s_ in stays_:
        idx_ = np.where(F4['stay_id'] == s_)[0]; A_ = BUN[s_]; mins = sorted(A_.minute.unique())
        for k_ in range(1, len(mins)):
            rows = A_[A_.minute == mins[k_]]
            if (rows.ekind != 1).any(): continue
            e_, u_, _, _ = bundle_vecs(rows)
            prev = X4[idx_[k_-1]]; dt_ = 60.0*(mins[k_]-mins[k_-1])
            hprev = prev[192:960].reshape(4, 192)*torch.exp(-dt_*FA)
            ctx = prev[:192] + LN(hprev.mean(0), 'core.out_norm') + mlpm(torch.tensor([np.log(dt_)/10], dtype=torch.float64), 'gap','gap')
            mine.append(mlpm(torch.cat([ctx, e_, u_]), 'z_post','zp')); prior_.append(mlpm(ctx, 'z_prior','zp')); true_.append(X4[idx_[k_], 960:992])
    return torch.stack(mine).numpy(), torch.stack(prior_).numpy(), torch.stack(true_).numpy()
ACT_E.update(enc='gelu', gap='gelu', zp='gelu')
mz, pz, tz = zpair(sub3)
print('true sd', tz.std().round(3), 'mine mean-part sd', mz[:,:32].std().round(3), 'prior mean sd', pz[:,:32].std().round(3))
for nm2, a2 in [('mine[:32]', mz[:,:32]), ('mine[32:]', mz[:,32:]), ('prior[:32]', pz[:,:32]), ('mine+prior', mz[:,:32]+pz[:,:32])]:
    c2 = np.corrcoef(a2.ravel(), tz.ravel())[0,1]; sl = np.polyfit(a2.ravel(), tz.ravel(), 1)[0]
    print(nm2, 'corr', round(c2,3), 'slope', round(sl,3), 'rmse', round(float(np.sqrt(((a2-tz)**2).mean())),3))
print('per-dim corr mine vs true (min/med)', np.round(np.percentile([np.corrcoef(mz[:,i],tz[:,i])[0,1] for i in range(32)],[0,50]),3))
```