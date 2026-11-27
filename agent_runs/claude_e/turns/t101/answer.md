Batch 0 shows the limit is not z noise. The lag-1 dose deviation decodes about equally from:
- the previous `out` (R² 0.28);
- the z-free context (0.26);
- `q` (0.21–0.22).

Lag 2 is about 0.05–0.10 everywhere. The full-state fit failed numerically and needs standardising.

The model's own lag-1 coefficient (−0.39) is about the true −2.04 times this R². So the model uses the dose information it carries correctly; it simply carries little of it. That makes the bottleneck representational.

One fixable source of noise: the deviation is the dose minus the protocol dose for the tier of the previous finger reading. If `q` encodes that reading imprecisely, the tier part of the dose leaks into the read-out as noise. But the head also gets the exact last-reading bin, through `embed.bins.value[lvb]` and the per-bin transition table. So per-bin offsets can subtract the tier dose exactly.

Next I'll test whether adding exact last-bin indicators, the true tier, or the head's 192 hidden activations raises the cross-validated R² for lag-1 deviation. I'll standardise the full state properly too. I'll also print the finger bin edges, to see whether they line up with the protocol thresholds 150/180/220/300.

If last-bin indicators lift R² to about 0.5, a unit-gain correction becomes affordable within the factual budget. If R² stays near 0.25, the correction must stay small; the per-hour factual cost is roughly coefficient² × read-out noise variance.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
print('gf edges:', np.round(np.array(VOCAB['bin_edges']['glucose_finger'] if 'bin_edges' in VOCAB else edges_gf),1) if 'bin_edges' in VOCAB else np.round(edges_gf,1))
ix = R5.index[R5.event=='glucose_finger']
# map Db0 rows to R5 rows (same stay,hour) to get lvb / zl
key5 = R5.loc[ix, ['stay','hour','lvb','zl','code']].copy(); key5['hour'] = key5.hour.round().astype(int); key5 = key5.rename(columns={'stay':'sid'})
Dd = Db0.merge(key5, on=['sid','hour'], how='left')
lvb_oh = np.eye(17)[Dd.lvb.values.astype(int)]
tier_oh = np.eye(5)[Dd.tier.values.astype(int)]
# hidden activations of value head at these rows (mean over z draws)
f_ = reg_feats(torch.as_tensor(Dd.has.values if 'has' in Dd else np.ones(len(Dd))), torch.as_tensor(np.full(len(Dd),3600.0)), torch.as_tensor(Dd.zl.values))
Q8_ = q_from_rows(Xs5t, prev, dtv5, nz=8, seed=1)
_, wc_, hid_ = value_logits(Q8_, torch.as_tensor(Dd.code.values.astype(int)), torch.as_tensor(Dd.lvb.values.astype(int)), f_)
HIDm = hid_.mean(0).numpy()
y1 = Dd.dev1.values.astype(float)
def cvr2(F_, y, std=False):
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1,5,16))) if std else RidgeCV(alphas=np.logspace(-2,4,14))
    pred = cross_val_predict(mdl, F_, y, cv=GroupKFold(5), groups=grp)
    return 1-np.mean((y-pred)**2)/y.var()
print('dev1 q+lvb onehot      :', round(cvr2(np.c_[Qg, lvb_oh], y1),3))
print('dev1 q+tier onehot     :', round(cvr2(np.c_[Qg, tier_oh], y1),3))
print('dev1 HID (mean z)      :', round(cvr2(HIDm, y1),3))
print('dev1 q+HID+lvb         :', round(cvr2(np.c_[Qg, HIDm, lvb_oh], y1),3))
print('dev1 full state (std)  :', round(cvr2(xs.numpy(), y1, std=True),3))
print('dev1 out+tier (std)    :', round(cvr2(np.c_[out_p.numpy(), tier_oh], y1, std=True),3))
# raw dose decodability within tier for comparison
D1raw = Dd.ins1.values.astype(float)
print('raw dose1 from q+tier: R2 of within-tier part', round(1-np.mean((D1raw - cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,14)), np.c_[Qg, tier_oh], D1raw, cv=GroupKFold(5), groups=grp))**2)/y1.var(),3))
```