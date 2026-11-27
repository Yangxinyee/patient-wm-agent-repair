The shortfalls are confirmed on rows where insulin ran in the previous hour: -1.60, -1.33 and -0.39 mg/dL per u/h of dose surprise at lags 1–3. The surprises have a standard deviation of about 0.65 u/h, and the predictive variance has a median of about 147 (sd about 12 mg/dL).

The honest, cross-validated decoders are weaker than my notes claimed:
- lag 1: R² 0.47, slope 0.39 at ridge alpha 1;
- lag 2: R² 0.20, slope 0.21.

Rough cost–benefit of an unbiased channel (decoder rescaled by 1/slope):
- **Lag 1.** About NLL-neutral. The added noise is about the same size as the corrected effect.
- **Lag 2.** Costs about 0.002 nats overall.

A cheaper route is a single lag-1 channel carrying the summed shortfall of about -3.3 per u/h. Under sustained dose edits it gives the same per-hour push as separate lag channels. In factual data it over-corrects lag 1 by about the lag-2 amount, which costs only about 0.0006 overall. Whether the model then carries the effect forward correctly in rollouts still has to be tested cleanly, since fwd 8 was corrupted.

First I want the best lag-1 decoder, so I will tune the ridge penalty downward. I will also measure the norepinephrine shortfalls and decoder quality on MAP rows with norepinephrine in the previous hour. A noticeably better R² at small alpha would make the unbiased channel affordable. If norepinephrine stays near R² 0.2, its channel must be weak or omitted.

```python
for al_ in [0.01, 0.1, 0.3]:
    pr_, _ = cv_ridge(mk, e1, al_)
    r2_ = 1 - np.nanmean((pr_[mk]-e1[mk])**2)/e1[mk].var(); k_ = np.polyfit(e1[mk], pr_[mk], 1)[0]
    print('eps_ins_1 alpha %g: cvR2 %.3f slope %.3f' % (al_, r2_, k_))
pres_n = (rd_.ne_1.fillna(0).values > 0); mkn = mapf & pres_n
D1, D2, D3 = [np.nan_to_num(rd_['D_ne_%d'%k].values) for k in (1,2,3)]
n1, n2, n3 = [np.nan_to_num(rd_['eps_ne_%d'%k].values) for k in (1,2,3)]
for nm_, X_ in [('D_ne', np.c_[D1,D2,D3]), ('eps_ne', np.c_[n1,n2,n3])]:
    cfn, *_ = np.linalg.lstsq(np.c_[X_[mkn], np.ones(mkn.sum())], res_[mkn], rcond=None)
    print(nm_, 'MAP deficits lag1-3 (n=%d):' % mkn.sum(), np.round(cfn[:3],2), 'sd', np.round(X_[mkn].std(0),3))
print('var_p map present median %.1f' % np.median(var_p[mkn]))
for tgt_nm, y_ in [('D_ne_1', D1), ('eps_ne_1', n1)]:
    for al_ in [0.1, 1.0]:
        pr_, _ = cv_ridge(mkn, y_, al_)
        r2_ = 1 - np.nanmean((pr_[mkn]-y_[mkn])**2)/y_[mkn].var(); k_ = np.polyfit(y_[mkn], pr_[mkn], 1)[0]
        print('%s alpha %g: cvR2 %.3f slope %.3f' % (tgt_nm, al_, r2_, k_))
```