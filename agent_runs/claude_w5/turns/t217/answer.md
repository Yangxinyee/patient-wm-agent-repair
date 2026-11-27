Two useful facts. The 4 slots are essentially copies of each other (they differ by 0.04 on a state of norm about 750), so one shared injection direction for all slots is fine. And I now have 2,393 drug bundles, each with x1, x2 and x½ versions of the u token and the prelude FFN input n3. 1,632 contain insulin, 2,219 norepinephrine, and 1,458 both.

Next I fit ridge decoders for the insulin and norepinephrine rates, with a split by stay. I compare n3 (prelude FFN input) against u (the scale-free action token). The test that matters is the slope of the decoded change against the true change when the dose is doubled or halved, plus any cross-talk into the other drug. I also measure how well the policy mean ĉ = E[rate | state] can be predicted from the pre-bundle state. A site with validation R² ≥ 0.95 and x2/x½ slopes near 1 can carry the signal. Sublinear slopes mean the rate signal needs linearizing first.

```python
sid_tr = set(sids_B[:70])
tr = np.array([r['sid'] in sid_tr for r in DB]); va = ~tr
YI = np.array([r['rI'] for r in DB]); YN = np.array([r['rN'] for r in DB])
def ridge_fit(X, y, lam):
    mu = X[tr].mean(0); Xc = X[tr]-mu; ym = y[tr].mean()
    w = np.linalg.solve(Xc.T@Xc + lam*np.eye(X.shape[1]), Xc.T@(y[tr]-ym)); return w, ym - mu@w
def r2(yh, y): return 1 - ((y-yh)**2).sum()/((y-y.mean())**2).sum()
dec = {}
for src in ('n3', 'u'):
    X1 = np.array([r[src][1.0] for r in DB])
    for tg, Y in (('I', YI), ('N', YN)):
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1, 10):
            w, b = ridge_fit(X1, Y, lam); s_ = r2(X1[va]@w+b, Y[va])
            if best is None or s_ > best[0]: best = (s_, lam, w, b)
        s_, lam, w, b = best; dec[(src, tg)] = (w, b)
        msg = f"{src} {tg}: val R2 {s_:.3f} lam {lam}"
        for f in (2.0, 0.5):
            key = (tg, f); idx = [i for i, r in enumerate(DB) if key in r[src]]
            dh = np.array([(DB[i][src][key] - DB[i][src][1.0]) @ w for i in idx])
            dt_ = np.array([(f-1)*(DB[i]['r'+tg]) for i in idx])
            sl = (dh@dt_)/(dt_@dt_); msg += f" | x{f}: slope {sl:.3f} R2 {r2(dh, dt_):.3f}"
        print(msg)
# cross-talk: N decoder under insulin x2
for src in ('n3','u'):
    w, b = dec[(src,'N')]; idx = [i for i, r in enumerate(DB) if ('I',2.0) in r[src]]
    dh = np.array([(DB[i][src][('I',2.0)] - DB[i][src][1.0]) @ w for i in idx]); dt_ = np.array([DB[i]['rI'] for i in idx])
    print(src, 'N-decoder change under ins x2: slope vs rI', round((dh@dt_)/(dt_@dt_),3), 'sd', round(dh.std(),3))
ST = np.array([r['st'] for r in DB])
for tg, Y in (('I', YI), ('N', YN)):
    for lam in (1, 10, 100):
        w, b = ridge_fit(ST, Y, lam); print('c_hat', tg, 'lam', lam, 'val R2', round(r2(ST[va]@w+b, Y[va]),3))
print('rate sd I', YI.std().round(3), 'N', YN.std().round(3), 'mean I', YI.mean().round(3), 'N', YN.mean().round(3))
```