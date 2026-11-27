The worst-case bound is far too loose: no z_post unit is provably dead at measurement bundles. The best candidates sit only around −4 at drug bundles. Rather than spend a forward pass on a probe, I will choose units by this ranking. The factual-fit request will then directly measure any cost of losing their contributions at measurement bundles.

The other ingredient is the expected dose given the model's context, E[rate|ctx]. It is subtracted so that only the dose surprise is injected in factual mode, which avoids double-counting the band-predictable effect. I fit it as a ridge regression of the recorded rate on ctx at drug-present bundles from the original model's states, with grouped-by-stay cross-validation. For insulin I compare against the policy's own predictor, the band mean of the latest glucose.

If ctx predicts insulin nearly as well as the band mean, a linear E[rate|ctx] is adequate. If it does much worse, I need nonlinear units or more data.

```python
from sklearn.model_selection import GroupKFold, cross_val_predict
d_idx = np.asarray(dr3); sidd = np.asarray(sid3)[d_idx]; hrd = np.asarray(hr3)[d_idx]
rI_map = {(s, round(h,3)): r for s, h, r in zip(tu.stay_id[tu.event=='insulin'], tu.hour[tu.event=='insulin'], tu.rate[tu.event=='insulin'])}
rN_map = {(s, round(h,3)): r for s, h, r in zip(tu.stay_id[tu.event=='norepinephrine'], tu.hour[tu.event=='norepinephrine'], tu.rate[tu.event=='norepinephrine'])}
RI_d = np.array([rI_map.get((s, round(h,3)), np.nan) for s, h in zip(sidd, hrd)]); RN_d = np.array([rN_map.get((s, round(h,3)), np.nan) for s, h in zip(sidd, hrd)])
print('drug bundles', len(d_idx), 'with ins', int((~np.isnan(RI_d)).sum()), 'with NE', int((~np.isnan(RN_d)).sum()), ' hour sample', np.round(hrd[:4],4))
C3 = np.asarray(CTX3, dtype=np.float64)
for nmz, R_ in [('INS', RI_d), ('NE', RN_d)]:
    mk = ~np.isnan(R_)
    for al in [1.0, 10.0, 100.0]:
        p_ = cross_val_predict(Ridge(alpha=al), C3[mk], R_[mk], groups=sidd[mk], cv=GroupKFold(5))
        print(nmz, 'alpha', al, 'CV R2 %.3f' % (1 - ((R_[mk]-p_)**2).mean()/R_[mk].var()))
```