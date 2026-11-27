The offline machinery checks out:
- wc_from at the prior-mean z reproduces WC0 exactly;
- HI['ob'] is the observed bin, and minus log of P0 at that bin gives nll0;
- aug_wc returns z-sampled with_code for chosen rows.

So I can score any candidate head offline, in fp32 and bf16, with sampled z.

Next I will build the insulin half of E3:
1. **Shortfall coefficients.** Re-estimate what the model misses on rows where insulin ran in the previous hour, by regressing the original model's glucose residual on the insulin surprise at lags 1–3.
2. **Per-row target tilt.** Convert the needed mean shift into a tilt by dividing by the row's predictive variance, with the tilt vector set to glucose centres divided by 100.
3. **Decoder and gates.** Fit a z-augmented ridge decoder of that tilt, plus the glucose-family and insulin-presence gates. Everything is cross-fitted by stay, so the reported R² and slope are honest.

I expect:
- lag-1 and lag-2 shortfalls near -1.6 and -1.3;
- a decoder R² of about 0.5 on present rows;
- gate separation near-perfect.
If the decoder is much weaker than that, the channel will be mostly noise, and I would cut its scale.

```python
from sklearn.linear_model import Ridge
rows_all = np.arange(len(rd_))
evv = rd_.event.values; sidv = rd_.stay_id.values
glu = np.isin(evv, ['glucose_finger','glucose_lab']); mapf = np.isin(evv, ['map_arterial','map_cuff'])
e1, e2, e3 = [np.nan_to_num(rd_['eps_ins_%d'%k].values) for k in (1,2,3)]
pres_i = (rd_.ins_1.fillna(0).values > 0)
res_ = rd_.observed.values - rd_.pred0.values
pp = P0[:,1:] / P0[:,1:].sum(1, keepdims=True)
mu_p = (pp*HI['cen']).sum(1); var_p = (pp*HI['cen']**2).sum(1) - mu_p**2
mk = glu & pres_i
Xd = np.c_[e1, e2, e3][mk]; cf_, *_ = np.linalg.lstsq(np.c_[Xd, np.ones(mk.sum())], res_[mk], rcond=None)
print('ins deficits lag1-3 (present glu rows, n=%d):' % mk.sum(), np.round(cf_[:3],2), ' eps sd', np.round(Xd.std(0),2))
print('var_p glu present: median %.0f, pct10/90 %.0f %.0f' % tuple(np.percentile(var_p[mk], [50,10,90])))
# z-aug data
ND = 3
XA = aug_wc(rows_all, ND, 11)   # (ND, n, 192)
ust = np.unique(sidv); fold = {s_: i_ % 5 for i_, s_ in enumerate(np.random.default_rng(0).permutation(ust))}
fv = np.array([fold[s_] for s_ in sidv])
def cv_ridge(mask, y, alpha=3.0):
    pred = np.full(len(y), np.nan); coefs = []
    for f_ in range(5):
        tr_ = mask & (fv != f_); te_ = mask & (fv == f_)
        Xtr = XA[:, tr_].reshape(-1, 192); ytr = np.tile(y[tr_], ND)
        m_ = Ridge(alpha=alpha).fit(Xtr, ytr)
        pred[te_] = m_.predict(WC0[te_])
    m_ = Ridge(alpha=alpha).fit(XA[:, mask].reshape(-1,192), np.tile(y[mask], ND))
    return pred, m_
for lag_, ek in [(1, e1), (2, e2)]:
    for al_ in [1.0, 10.0]:
        pr_, _ = cv_ridge(mk, ek, al_)
        r2_ = 1 - np.nanmean((pr_[mk]-ek[mk])**2)/ek[mk].var()
        k_ = np.polyfit(ek[mk], pr_[mk], 1)[0]
        print('decoder eps_ins_%d alpha %g: cvR2 %.3f slope %.3f' % (lag_, al_, r2_, k_))
```