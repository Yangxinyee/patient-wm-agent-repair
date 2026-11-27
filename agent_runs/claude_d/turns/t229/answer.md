Pre-rounding removed the systematic bf16 bias: it is now −0.02 to −0.06 mg/dL, with a random spread of about 0.45. κ = 6 with either gate penalty costs only +0.0005 NLL at the strong setting, C = −4.92. κ = 10 costs more, probably because its larger weights carry larger rounding errors.

There is a new warning sign, though. The lag-1 decoder's weights are huge: RMS 7.6 per input, norm about 105. Its output is a near-cancelling sum of large terms. My bf16 check only rounds the final head input. In real rollouts the head input is itself computed through the whole recurrent state in bf16, with perhaps about 1% relative error. That could move such a decoder by about 1 u/h, and a small systematic part would bring back a drift of several mg/dL.

The original head's rollout at hour 13 agreed with teacher forcing to 0.2 mg/dL, which shows how robust a well-conditioned head is. I want the same for E3.

So I will trade a little decoder R² for a much smaller norm, comparing:
- ridge penalties from 0.01 to 1;
- PCA-truncated fits.

For each I will report held-out R², slope, weight norm, and the output spread under random 1% relative perturbations of the head input. The aim is a decoder whose sensitivity is well below 0.1 u/h while keeping R² around 0.5 or more.

```python
from numpy.linalg import svd
rng_ = np.random.default_rng(5)
mtr_ = trm & mk; mte_ = tem & mk
Xtr_ = XA[:, mtr_].reshape(-1,192); ytr_ = np.tile(e1[mtr_], ND)
pert = WC0[mte_] * (1 + 0.01*rng_.standard_normal(WC0[mte_].shape))
def rep(nm_, w_, b_):
    p_ = WC0[mte_] @ w_ + b_
    r2_ = 1 - np.mean((p_-e1[mte_])**2)/e1[mte_].var(); k_ = np.polyfit(e1[mte_], p_, 1)[0]
    sens = np.std(pert @ w_ + b_ - p_)
    print('%-12s R2 %.3f slope %.3f |w| %6.1f  sens(1%%) %.3f  R2/slope-unbiased noise-ratio %.2f' % (nm_, r2_, k_, np.linalg.norm(w_), sens, np.sqrt(max(1-(1-k_)**2-r2_,1e-9))/k_))
for al_ in [0.01, 0.1, 1.0, 10.0, 100.0]:
    m_ = Ridge(alpha=al_).fit(Xtr_, ytr_); rep('ridge %g' % al_, m_.coef_, m_.intercept_)
mu_ = Xtr_.mean(0); U_, S_, Vt_ = svd(Xtr_ - mu_, full_matrices=False)
for nc_ in [16, 32, 64, 96]:
    Z_ = (Xtr_ - mu_) @ Vt_[:nc_].T
    m_ = Ridge(alpha=1e-6).fit(Z_, ytr_)
    w_ = Vt_[:nc_].T @ m_.coef_; b_ = m_.intercept_ - mu_ @ w_
    rep('pca %d' % nc_, w_, b_)
print('wc component sd range', np.round(WC0.std(0).min(),3), np.round(WC0.std(0).max(),3), 'mean |wc|', np.round(np.abs(WC0).mean(),3))
```