The C = 3 build is badly broken: overall NLL rises by +0.375. The gated rows (+0.067) and arterial no-norepinephrine rows (+0.003) are far too small to explain that, so the damage is on other rows.

The likely cause is extrapolation. I fitted the decoder only on gated rows. On rows of other categories its output can be large, so the signal s overwhelms the gate margin of κ·0.5 = 3 and the unit opens with a big tilt. My tilt scale (lam = 1) also forced s to be large.

Two fixes:
1. Raise lam to 4, so that s stays about ±0.3 per standard deviation.
2. Fit the decoder with zero targets on the risky off-gate rows, those with G ≥ 0.6, so its output stays small there.

This run first shows the current decoder's off-gate magnitudes by category. It then refits with zero-target weights ω of 0.1, 0.5 and 2, and evaluates the C = 6 build per category. The fix is confirmed if off-gate damage disappears while the gated rows keep cvR2 near 0.18.

```python
cats = {'A&ne':gate_rows,'A&no':noA,'C&ne':isC&(ne1>0),'C&no':isC&(ne1<=0),'g&ne':(~isM)&(ne1>0),'g&no':(~isM)&(ne1<=0)}
Dmax = np.abs(WCd@wd + bd).max(0); Gmx = Gd.max(0)
print('raw dec |D|max q50/q99/q999 by cat:', {nm: list(np.round(np.quantile(Dmax[m_], [.5,.99,.999]),2)) for nm,m_ in cats.items()})
def ridge_fit_w(X, y, sw, a):
    Xm = (sw[:,None]*X).sum(0)/sw.sum(); ym = (sw*y).sum()/sw.sum(); Xc = X - Xm
    w = solve((Xc*sw[:,None]).T@Xc + a*np.eye(X.shape[1]), (Xc*sw[:,None]).T@(y-ym)); return w, ym - Xm@w
off = np.where((~gate_rows) & (Gmx>0.6))[0]
def fitdec(gtr, otr, om, a=0.1):
    rows_ = np.r_[gtr, otr]; y_ = np.r_[Dn[gtr], np.zeros(len(otr))]; sw_ = np.r_[np.ones(len(gtr)), om*np.ones(len(otr))]
    return ridge_fit_w(WCd[:, rows_, :].reshape(-1,192), np.tile(y_, nd), np.tile(sw_, nd), a)
for om in [0.1, 0.5, 2.0]:
    p = np.zeros(len(gr))
    for tr, te in GroupKFold(5).split(gr, groups=sid[gr]):
        w, b = fitdec(gr[tr], off[~np.isin(sid[off], sid[gr[te]])], om); p[te] = (WCd[:, gr[te], :]@w + b).mean(0)
    r2 = 1-((yD-p)**2).mean()/yD.var(); kk = np.cov(p,yD)[0,1]/yD.var()
    w2_, b2_ = fitdec(gr, off, om); Dm2 = np.abs(WCd@w2_ + b2_).max(0)
    Wn = build_ne(6.0, w2_, b2_, kk, lam=4.0); r = eval_E(*Wn); dn = r[0]-r0[0]
    Dh2 = (WCd[:, gr, :]@w2_ + b2_).mean(0)/kk; sl = np.polyfit(Dh2, (r[1]-rE[1])[gr], 1)[0]
    print('om', om, 'cv', round(r2,3), round(kk,3), 'offmax q999', round(float(np.quantile(Dm2[off], .999)),2), 'dAll', round(float(dn.mean()),5), {nm: round(float(dn[m_].mean()),4) for nm,m_ in cats.items()}, 'slope', round(sl,2))
```