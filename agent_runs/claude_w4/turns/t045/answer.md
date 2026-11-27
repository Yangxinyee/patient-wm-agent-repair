The factual CSV has, per reading, the observed value, the predicted mean and the NLL. The dose_edit CSV has, per stay-hour, the shift and a `reached` flag. My kernel still holds the train and tuning panels (`g`, `gT`), the insulin expected rate by glucose band (`band_mean`), and the impulse responses from the records (`bF` for finger, `bL` for lab, `bA` for arterial MAP).

Now the key diagnostic. For the 100 tuning stays in fwd_2, I compute each hour's insulin surprise: the recorded rate minus E[rate | sliding-scale band of the latest glucose]. I then regress the old model's factual residuals (observed − predicted mean) for glucose_finger and glucose_lab on the surprises at lags 1–8.

- **Old model ignores doses:** the lag-1 coefficient is near −1.8, matching the true one-hour effect, and later lags are smaller than the impulse response because intervening readings absorb part of the effect. Then a surprise-based edit both fixes the counterfactuals and slightly improves the factual NLL.
- **Old model already handles in-band surprises:** coefficients near 0. Its failure is then only extrapolation to ×2 doses.

```python
ids100 = sorted(fo.stay_id.unique()); print('n stays in factual:', len(ids100), 'same as sel100:', set(ids100)==set(sel100))
T = gT[gT.stay_id.isin(ids100)].sort_values(['stay_id','h']).copy()
def band_np(v):
    b = np.zeros(len(v), int); b[v>150]=1; b[v>180]=2; b[v>220]=3; b[v>260]=4; b[np.isnan(v)]=0; return b
lat = []
for sid, grp in T.groupby('stay_id', sort=False):
    cur = np.nan; out_ = []
    for f_, l_ in zip(grp.glucose_finger.values, grp.glucose_lab.values):
        if not np.isnan(f_): cur = f_
        elif not np.isnan(l_): cur = l_
        out_.append(cur)
    lat += out_
T['latG'] = lat; T['band'] = band_np(T.latG.values)
r = T.ins_rate.fillna(0).values
bm = band_mean.reindex(range(5)).fillna(0).values; print('band_mean', np.round(bm,3))
T['s_ins'] = r - bm[T.band.values]
print('rate>0 with band0:', ((r>0)&(T.band.values==0)).sum(), ' band>0 with no dose:', ((r==0)&(T.band.values>0)).sum())
key = T.set_index(['stay_id','h']).s_ins
fo['R'] = fo.observed - fo.predicted_mean
K=8
for ev in ['glucose_finger','glucose_lab']:
    d = fo[fo.event==ev].copy()
    X = np.column_stack([key.reindex(list(zip(d.stay_id, d.hour - j))).fillna(0).values for j in range(1,K+1)])
    A = np.column_stack([np.ones(len(d)), X]); yv = d.R.values
    bcoef, *_ = np.linalg.lstsq(A, yv, rcond=None); resid = yv - A@bcoef
    cov = np.linalg.inv(A.T@A) * resid.var(); se_ = np.sqrt(np.diag(cov))
    print(ev, 'n', len(d), 'mean R %.2f sd %.2f' % (yv.mean(), yv.std()))
    print('  gamma', np.round(bcoef[1:],2)); print('  se   ', np.round(se_[1:],2))
```