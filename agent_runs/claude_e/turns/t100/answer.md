Batch 0 gives a consistent picture of the model's dose coefficients but a problem with reading the dose out.

**Model coefficients** (lags 1–6) are [−0.39, +0.24, +0.08, +0.11, +0.22, +0.34], with standard errors of about 0.10. This matches the first tuning estimate and is far from the true [−2.04, −1.23, −0.25, +0.38, +0.80, +0.69]. So the per-lag gap Δβ ≈ [−1.65, −1.47, −0.33, +0.27, +0.58, +0.35] looks real. The model nearly ignores the within-tier dose for 1–2 hours afterwards.

**The problem:** a linear read-out of the dose deviation from the head's query vector `q` is weak. Cross-validated R² is 0.21 at lag 1 and about 0 at lags 2 and beyond. A correction built on that would be noisy in factual data and blind to older doses.

One possible cause is the latent z: each z draw adds `z_head(z)` noise to the view before normalisation, which could hide a dose signal that is present in the state. I'll compare decodability from four sources:
- the previous bundle's `out`;
- the z-free context (`out` + normalised slot mean + gap);
- `q` averaged over z draws;
- the full previous state.

If the z-free context decodes dev1 much better (R² ≥ 0.5), a read-out direction orthogonal to the `z_head` subspace can recover it inside the head. If even the full state decodes it poorly at the reading time, the head-only plan cannot work, and I need to change how the state carries the dose.

```python
Xs5t = Xs5 if torch.is_tensor(Xs5) else torch.as_tensor(np.asarray(Xs5))
prev = torch.as_tensor(Db0.prev.values.astype(int)); dtv5 = torch.as_tensor(Db0.dt.values, dtype=torch.float64)
xs = Xs5t[prev]; out_p = xs[:, :192]; h4p = xs[:, 192:960].reshape(-1,4,192)
ctx5 = ctx_from_state(out_p, h4p, dtv5)
feats = {
  'out_prev': out_p.numpy(),
  'ctx(no z)': ctx5.numpy(),
  'q mean z': Qg,
  'full state prev': xs.numpy(),
}
# q computed from ctx with z set to prior mean (no noise)
o = mlp2(ctx5, 'z_prior'); mu = o[:, :32]
q_mu = ln(ctx5 + lin(mu, 'z_head') + T('type_emb.weight')[1], 'query_norm').numpy()
feats['q (z=prior mean)'] = q_mu
# also the state right after the insulin bundle (if the prev bundle is it) - indicator
St5 = np.load('/work/fwd_5/states_original.npz', allow_pickle=True)
hasins_prev = np.asarray(St5['has_insulin'])[Db0.prev.values.astype(int)]
print('frac rows whose prev bundle has insulin:', hasins_prev.mean())
for k in [1, 2]:
    y = Db0[f'dev{k}'].values.astype(float)
    for nm, F_ in feats.items():
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,14)), F_, y, cv=GroupKFold(5), groups=grp)
        print(f'dev{k} from {nm:18s}: CV R2 {1-np.mean((y-pred)**2)/y.var():.3f}')
```