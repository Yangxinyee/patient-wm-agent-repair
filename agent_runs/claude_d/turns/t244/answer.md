`eval_E` returns two per-reading arrays, probably NLL and predicted mean; I will confirm that against nll0/pred0 when I use it. WCd is the 4 z-draw head input (its mean correlates 0.975 with WC0). There are 4,931 arterial and 6,306 cuff MAP readings.

The planned norepinephrine channel has three parts:
- An AND-gate: "arterial MAP" plus "norepinephrine ran in the previous hour", built from linear gates that must have clean margins on all 4 z-draws.
- A decoder of the dose surprise D_ne_1 on the gated rows.
- A gentle exponential tilt of the MAP bins, with small amplitude so bf16 rounding cannot dominate.

I will gate on arterial only. That is the readout code, and an exponential tilt would shift cuff rows twice as much because their predictive variance is larger.

This run checks the two pieces that decide feasibility:
1. Gate margins. Gated rows should have G well above 1.5 and all other rows well below it.
2. The z-augmented decoder's cross-validated R2 and slope for D_ne_1.

I will also compute the model's deficit on cuff rows for reference.

```python
from numpy.linalg import solve
from sklearn.model_selection import GroupKFold
ev = RD2.event.values; isA = ev=='map_arterial'; isC = ev=='map_cuff'; isM = isA|isC
ne1 = RD2.ne_1.values; Dn = RD2.D_ne_1.values; sid = RD2.stay_id.values
gate_rows = isA & (ne1>0)
def ridge_fit(X, y, a):
    Xm = X.mean(0); ym = y.mean(); Xc = X - Xm
    w = solve(Xc.T@Xc + a*np.eye(X.shape[1]), Xc.T@(y-ym)); return w, ym - Xm@w
nd = WCd.shape[0]; Xall = WCd.reshape(-1,192)
wf, bf = ridge_fit(Xall, np.tile(isA.astype(float), nd), 1.0)
wp, bp = ridge_fit(Xall, np.tile((ne1>0).astype(float), nd), 1.0)
Gd = np.stack([WCd[d]@(wf+wp) + bf+bp for d in range(nd)])
for nm, msk in [('A&ne',gate_rows),('A&no',isA&(ne1<=0)),('C&ne',isC&(ne1>0)),('C&no',isC&(ne1<=0)),('glu',~isM)]:
    print(nm, int(msk.sum()), np.round(np.quantile(Gd[:,msk], [.001,.01,.5,.99,.999]),2))
gr = np.where(gate_rows)[0]
for a in [0.3, 3.0, 30.0]:
    p = np.zeros(len(gr))
    for tr, te in GroupKFold(5).split(gr, groups=sid[gr]):
        w, b = ridge_fit(WCd[:, gr[tr], :].reshape(-1,192), np.tile(Dn[gr[tr]], nd), a)
        p[te] = (WCd[:, gr[te], :]@w + b).mean(0)
    y = Dn[gr]; print('dec a', a, round(1-((y-p)**2).mean()/y.var(),3), 'k', round(np.cov(p,y)[0,1]/y.var(),3))
cm = isC&(ne1>0); print('cuff deficit slope', round(np.polyfit(Dn[cm], res[cm], 1)[0],2), int(cm.sum()))
```