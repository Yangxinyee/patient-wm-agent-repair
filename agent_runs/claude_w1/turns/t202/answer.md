The DEC ridge models have 576 coefficients. That is most likely a three-output ridge (dose lags 1–3) on the 192-dim head input, with coef_ shaped (3, 192). I'll take the lag-1 row and run the same diagnostic. Response ≈ 0.86 to the edits with sd < 1 would point to a bug in my minimum-variance solver. A much smaller response means the edit regime really differs from the perturbation regime the decoders were trained on.

```python
for dk,oc_code,keys in [('glucose_finger',0,keysI+keysN),('map_arterial',2,keysN+keysI)]:
    co_=np.asarray(DEC[dk].coef_); print(dk, co_.shape)
    w_=co_.reshape(-1,192)[0]; rows_=np.where(cf==oc_code)[0]
    print(' lag1 sdw', (Rf[rows_]@w_).std().round(3), 'sdz', (DQf[rows_,0,:].astype(np.float64)@w_).std().round(3), 'resp', diag(w_, rows_, oc_code, keys)[2])
    print(' corr(dec, true lag1 dose) on factual rows:', end=' ')
    # true lag-1 dose per factual row of this code
    dl=[]; 
    for ii_,g_ in enumerate(GEN):
        dose_=g_['ins'] if oc_code==0 else g_['ne']; hh_=g_['hour'].astype(int); jj=np.clip(hh_,0,26)
        dl.append(np.where(hh_>=0, dose_[jj], 0.0))
    dl=np.concatenate(dl); pr_=Xf[rows_]@w_+DEC[dk].intercept_.ravel()[0]
    print(np.corrcoef(pr_, dl[rows_])[0,1].round(3), 'slope', np.polyfit(dl[rows_], pr_,1)[0].round(3), 'err sd', (pr_-dl[rows_]).std().round(3))
```