The table norms explain why my variants made no difference:
- **Row 0 is zero in every categorical table**, so absent fields add nothing.
- **Rows near 0.26–0.29 in norm are effectively untrained.**
- **`embed.kq` row 0 (0.71) and all three `embed.kind` rows** are the only possibly-missing constant terms, and they barely changed the error.

So the gap is probably not a missing embedding term, but something structural. The context, the encoder layout or the head might be slightly different.

Before guessing further, this run compares my posterior means with the saved ones dimension by dimension. It reports correlation, regression slope and offset for each dimension, using measurement-only bundles from all 99 stays.

- **High correlation with a slope ≠ 1 or an offset:** a scale or normalisation detail is wrong.
- **Correlation that is also low:** my e_k or context is missing real information.

I also check whether the error grows with the number of atoms in the bundle, which would point at the set encoder's size term or the 1/√n scaling.

```python
cand2 = []
for idx_ in st_s.index:
    r_ = st.loc[idx_]
    if r_.hi or r_.hn or r_.hour < 0: continue
    g_ = ev_by.get((r_.stay, r_.hour))
    if g_ is None or not g_.event.isin(list(CEN)).all(): continue
    ss_ = st_s[(st_s.stay == r_.stay) & (st_s.hour < r_.hour)]
    if len(ss_) == 0: continue
    cand2.append((idx_, ss_.index[-1], (r_.hour - ss_.hour.values[-1]) * 3600, g_))
    if len(cand2) >= 400: break
ZP, ZT, ZPR, NA = [], [], [], []
for idx_, ip, dt, g_ in cand2:
    vecs = [base_mk(e_, v_) for e_, v_ in zip(g_.event, g_.value)]
    e_k = set_enc(vecs, 'encode'); c_ = ctx_of(ip, dt)
    ZP.append(mlp(np.concatenate([c_, e_k, np.zeros(192)]), 'z_post')[:32]); ZPR.append(mlp(c_, 'z_prior')[:32])
    ZT.append(X[idx_, 960:992]); NA.append(len(g_))
ZP, ZT, ZPR, NA = map(np.array, (ZP, ZT, ZPR, NA))
cor = [np.corrcoef(ZP[:, j], ZT[:, j])[0, 1] for j in range(32)]
slope = [np.polyfit(ZP[:, j], ZT[:, j], 1)[0] for j in range(32)]
corp = [np.corrcoef(ZPR[:, j], ZT[:, j])[0, 1] for j in range(32)]
print('n', len(ZP), 'per-dim corr (post):', np.round(cor, 2))
print('per-dim corr (prior):', np.round(corp, 2))
print('slopes:', np.round(slope, 2))
print('sd saved', np.round(ZT.std(0)[:8], 3), 'sd mine', np.round(ZP.std(0)[:8], 3))
for n_ in np.unique(NA): print('n atoms', n_, 'count', (NA == n_).sum(), 'err %.4f' % np.abs(ZP[NA == n_] - ZT[NA == n_]).mean())
```