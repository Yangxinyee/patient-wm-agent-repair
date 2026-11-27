The spread is not the problem: predicted sd barely changes (glucose -0.002 on 13, MAP +0.0003 on 5). The whole NLL rise sits in the top quartile of shift size: +0.0067 for glucose (shifts 0.7–5.4 mg/dL) and +0.011 for MAP (0.33–2.3 mmHg, against a predictive sd of about 5). Smaller shifts are neutral or slightly helpful.

So large shifts are where the injection does not pay off. Either they occur where the decoded residual is wrong (decoder extrapolation, or a bad linear policy mean such as at infusion starts), or the binned NLL penalizes shifts more steeply than a Gaussian would.

Next I fit, per row, ΔNLL ≈ -a·(err·shift) + b·shift², where err = observed − original mean. I also check, within the top quartile, whether the error still follows the shift. A slope near 1 there, with b much larger than a, means the binned likelihood is unusually sensitive. A slope near 0 for large shifts means those residuals are spurious and should be clipped or down-weighted.

```python
for c in ('glucose_finger', 'map_arterial', 'map_cuff', 'glucose_lab'):
    m = codes_[ia] == c; e_ = err0[ia][m]; d_ = dm[ia][m]; dn = (n1-n0)[ia][m]
    A = np.c_[e_*d_, d_**2]; co = np.linalg.lstsq(A, dn, rcond=None)[0]
    q75 = np.quantile(np.abs(d_), 0.75); t = np.abs(d_) >= q75
    sl_top = np.cov(e_[t], d_[t])[0,1] / d_[t].var(); sl_low = np.cov(e_[~t], d_[~t])[0,1] / d_[~t].var()
    mse_ch = ((e_-d_)**2 - e_**2).mean()
    print(c, 'dNLL ~ -a*err*shift + b*shift^2: a', (-co[0]).round(5), 'b', co[1].round(5), '| 1/sd^2', round(1/np.mean(sd0[torch.from_numpy(codes_[ia_m]==c)].numpy()**2),5),
          '| slope top q', round(sl_top,3), 'low', round(sl_low,3), '| MSE change', mse_ch.round(4))
# which bundles produce the largest MAP shifts? characterize by x values and rates
m = codes_[ia] == 'map_arterial'; d_ = dm[ia][m]; X_ = XA[m]; t = np.abs(d_) >= np.quantile(np.abs(d_), 0.9)
print('top-10% MAP shift rows: xN mean', X_[t,1].mean().round(3), 'rN mean', X_[t,3].mean().round(3), '| all: xN sd', X_[:,1].std().round(3), 'rN mean', X_[:,3].mean().round(3))
print('frac with xN>0 in top', (X_[t,1] > 0).mean().round(3), '| err mean in top', err0[ia][m][t].mean().round(3), 'shift mean in top', d_[t].mean().round(3))
```