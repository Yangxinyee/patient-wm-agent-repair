The probe results show how far the state encodes the doses, as measured at the value-head input (stay-grouped CV):
- Insulin's raw recent dose is decoded fairly well (lag 1 R² 0.77, falling to 0.65 at lag 3). The insulin dose shock (dose minus policy-expected dose) is decoded much worse: R² 0.41 at lag 1, 0.14 at lag 2.
- Norepinephrine is weaker still: raw rate R² 0.63, shock R² only 0.12.
- Smaller ridge penalties help.

Only the shocks give an unconfounded estimate of how the state responds to a dose change. A correction that responds one-for-one to real dose changes must therefore use the shock probe scaled up by roughly 1/0.37 (insulin) or 1/0.12 (norepinephrine). That adds decoding noise to every factual prediction. Rough arithmetic suggests the insulin cost is modest but the norepinephrine cost may break the 0.005-nat budget.

Rather than trust formulas, I will measure this directly on the 99 stays where I have the tool's factual results:
- fit the shock decoders on the 300 other stays;
- rescale them to unit gain using the held-out slope;
- add the corresponding mean shift (as a logit tilt) to my reconstructed value head;
- compute the change in fine-bin NLL for a grid of gains.

I will also check that my bin convention reproduces the tool's per-reading NLL, and regress the tool's errors on the decoded shock to get the factual-optimal gain. I expect insulin gains of about −1.5 to −2 mg/dL per unit to cost at most ~0.002 nats. If norepinephrine at ~+3 mmHg per unit costs more than ~0.005, it needs a better decoder or a different construction.

```python
import numpy as np
from sklearn.linear_model import Ridge
edges_v = {e_['code']: np.array(e_['fine_bin_edges'], dtype=float) for e_ in VOCAB['numeric'] if e_['attribute']=='value'}
cname = {5:'map_cuff',6:'glucose_finger',7:'map_arterial',10:'glucose_lab'}
cen_v = {k_: np.array(v_, dtype=float) for k_, v_ in VOCAB['value_bin_centres'].items()}
c99 = codec[:5521]; vals99 = M4['value'].values.astype(float)
for side in ['left','right']:
    yb = np.array([1 + np.searchsorted(edges_v[cname[c]], v, side=side) for c, v in zip(c99, vals99)])
    nr = -np.log(p99[np.arange(5521), yb])
    print(side, "corr recon nll vs tool %.4f  mean recon %.4f tool %.4f" % (np.corrcoef(nr, M4['nll'].values)[0,1], nr.mean(), M4['nll'].mean()))
ybin99 = np.array([1 + np.searchsorted(edges_v[cname[c]], v, side='left') for c, v in zip(c99, vals99)])
lg99 = Hact99 @ W2.T + b2 + TR99
def nll_of(lg, yb):
    l_ = lg - lg.max(1, keepdims=True); lp = l_ - np.log(np.exp(l_).sum(1, keepdims=True)); return -lp[np.arange(len(yb)), yb]
base_nll = nll_of(lg99, ybin99)
tiltG = np.r_[0.0, (cen_v['glucose_finger'] - 160.0)/100.0]
tiltM = np.r_[0.0, (cen_v['map_arterial'] - 70.0)/20.0]
def cov_tilt(pp, cen, tilt):
    m1 = (pp[:,1:]*cen).sum(1); t1 = (pp*tilt).sum(1); return (pp[:,1:]*cen*tilt[1:]).sum(1) - m1*t1
cen_row = np.array([cen_v[cname[c]] for c in c99])
covG = np.array([cov_tilt(p99[i:i+1], cen_row[i], tiltG)[0] for i in range(5521)])
covM = np.array([cov_tilt(p99[i:i+1], cen_row[i], tiltM)[0] for i in range(5521)])
isG99 = np.isin(c99, [6,10]); isM99 = np.isin(c99, [5,7]); tr_ = np.arange(len(codec)) >= 5521
print("mean tilt-cov: glucose rows %.2f (sd %.2f), MAP rows %.2f (sd %.2f)" % (covG[isG99].mean(), covG[isG99].std(), covM[isM99].mean(), covM[isM99].std()))
res_dec = {}
for nm_, Mx, cs in [('ins', eps_ins, [6,10]), ('ne', eps_ne, [5,7])]:
    y = lagged(Mx, 1); mtr = tr_ & np.isin(codec, cs); mte = (~tr_) & np.isin(codec, cs)
    rg = Ridge(alpha=0.3).fit(WCc[mtr], y[mtr]); pr_ = rg.predict(WCc[:5521])
    sl = np.cov(pr_[mte[:5521] if False else np.isin(c99, cs)], y[:5521][np.isin(c99, cs)])[0,1] / np.var(y[:5521][np.isin(c99, cs)])
    res_dec[nm_] = (rg, sl)
    su = (pr_ - rg.intercept_*0) / sl
    mrow = np.isin(c99, cs)
    e_ = err4[mrow]; s_true = y[:5521][mrow]
    kf = np.cov(su[mrow], e_)[0,1]/np.var(su[mrow]); kt = np.cov(s_true, e_)[0,1]/np.var(s_true)
    print(f"{nm_}: held-out slope {sl:.3f}; err-regression slope on unit-gain probe {kf:.2f}, on true shock {kt:.2f}; probe sd {su[mrow].std():.3f} vs shock sd {s_true.std():.3f}")
    covx = covG if nm_=='ins' else covM; tilt = tiltG if nm_=='ins' else tiltM
    for kap in ([-1.0,-2.0,-3.0,-4.0] if nm_=='ins' else [1.0,2.0,3.0,4.0]):
        beta = kap*su/covx[mrow].mean()
        lg2 = lg99.copy(); lg2[mrow] += beta[mrow][:,None]*tilt[None,:]
        dn = nll_of(lg2, ybin99) - base_nll
        print(f"   kappa {kap:+.1f}: dNLL on its rows {dn[mrow].mean():+.5f}, per reading overall {dn.mean():+.5f}")
```