I already had what norepinephrine needs. `pred_ne` predicts the rate from MAP-history features only (R² 0.76). Its residual `nu_ne` is the level deviation:
- sd 0.25 on hours with norepinephrine running, against 0.15 for the one-step shock;
- strongly persistent (lag-1 autocorrelation about 0.82).

This is the part of the rate that the MAP history does not explain. A sustained edit such as "x2 from hour 12" moves exactly this quantity, hour after hour. For insulin, rates are redrawn every hour, so the one-step shock is already the level deviation.

This run tests the norepinephrine level-deviation target the same way as before:
- fit a decoder on the 300 other stays and measure its held-out R² and slope;
- regress the tool's errors on the true level deviation (factual-optimal gain) and on the unit-gain decoder;
- compute the factual NLL cost of unit-gain corrections with gains +1…+4 mmHg per unit.

I expect better decodability than the one-step shock (R² well above 0.12). I also expect a large positive error slope on the true deviation (the model ignores most of the rate's effect), which would make a correction with gain around +2 to +3 cheap or even helpful factually. If the decoder is still mostly noise, MAP needs a different construction.

```python
import numpy as np
from sklearn.linear_model import Ridge
mMc = np.isin(codec, [5, 7]); mGc = np.isin(codec, [6, 10])
def eval_decoder(nm_, y, mask_c, rows99, kgrid, tilt, covx, alpha=0.3):
    mtr = tr_ & mask_c
    rg = Ridge(alpha=alpha).fit(WCc[mtr], y[mtr])
    pr_ = rg.predict(WCc[:5521]); yy_ = y[:5521]
    sl = np.cov(pr_[rows99], yy_[rows99])[0,1]/np.var(yy_[rows99])
    r2 = 1 - ((yy_[rows99]-pr_[rows99])**2).sum()/((yy_[rows99]-yy_[rows99].mean())**2).sum()
    su = pr_/sl
    e_ = err4[rows99]
    kt = np.cov(yy_[rows99], e_)[0,1]/np.var(yy_[rows99]); kf = np.cov(su[rows99], e_)[0,1]/np.var(su[rows99])
    print(f"{nm_}: held-out R2 {r2:.3f} slope {sl:.3f} | err slope on true {kt:.2f}, on unit-gain probe {kf:.2f} | probe sd {su[rows99].std():.3f}, target sd {yy_[rows99].std():.3f}")
    out_ = {}
    for kap in kgrid:
        beta = kap*su/covx[rows99].mean()
        lg2 = lg99.copy(); lg2[rows99] += beta[rows99][:,None]*tilt[None,:]
        dn = nll_of(lg2, ybin99) - base_nll
        out_[kap] = (dn[rows99].mean(), dn.mean())
        print(f"   kappa {kap:+.1f}: dNLL its rows {dn[rows99].mean():+.5f}, overall {dn.mean():+.5f}")
    return rg, sl, su
rows99M = np.isin(c99, [5, 7]); rows99G = np.isin(c99, [6, 10])
dec = {}
for lag in [1, 2]:
    y = lagged(nu_ne, lag)
    dec[('nelvl', lag)] = eval_decoder(f"ne level dev lag{lag}", y, mMc, rows99M, [1.0, 2.0, 3.0, 4.0, 6.0], tiltM, covM)
y_ = lagged(nu_ne, 1); e_ = err4[rows99M]
X_ = np.c_[lagged(nu_ne,1)[:5521][rows99M], lagged(eps_ne,1)[:5521][rows99M], np.ones(rows99M.sum())]
cf_ = np.linalg.lstsq(X_, e_, rcond=None)[0]
print("err on [level dev lag1, one-step shock lag1]:", np.round(cf_[:2], 2))
for alpha in [0.03, 0.3, 3.0]:
    rg = Ridge(alpha=alpha).fit(WCc[tr_ & mMc], lagged(nu_ne,1)[tr_ & mMc]); pr_ = rg.predict(WCc[:5521])
    yy_ = lagged(nu_ne,1)[:5521]
    print("alpha", alpha, "held-out R2 %.3f" % (1 - ((yy_[rows99M]-pr_[rows99M])**2).sum()/((yy_[rows99M]-yy_[rows99M].mean())**2).sum()))
```