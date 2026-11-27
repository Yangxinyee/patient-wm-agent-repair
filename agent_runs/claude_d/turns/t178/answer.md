The decoders are too weak to rely on. From with_code, the insulin surprise has a cross-validated R² of 0.30 at lag 1, 0.17 at lag 2 and 0.05 at lag 3. The NE surprise has only 0.09, 0.04 and 0.01. The decoded values respond to the true surprise with slopes equal to these R² values.

To get the full counterfactual response I would have to scale them up by 1/R². That would add enough noise to factual predictions to break the NLL budget, mainly for NE. Rough estimates:

- NE at lag 1 alone: about +0.009 nats overall.
- insulin at lag 2: about +0.003.

So I need a richer readout of the dose signal. Next I compare the cross-validated decodability of the lag-1 and lag-2 surprises from several inputs:

1. with_code at a lower ridge penalty;
2. the head's hidden activations, which a value_bin.2 edit could read;
3. both together;
4. the full 992-number state (upper bound).

If the hidden activations or the full state raise R² well above 0.3 (insulin) and 0.1 (NE), I will move the edit to where that signal is readable. Otherwise I need an upstream edit.

```python
def cvr2(X, Y, grp, alphas=(0.3,1,3,10,30)):
    mu = X.mean(0); sd = X.std(0)+1e-6; Xs=(X-mu)/sd
    best=-9; bk=None
    for a in alphas:
        Yh = np.zeros_like(Y)
        for trn,tst in GroupKFold(5).split(Xs,Y,grp):
            Yh[tst]=Ridge(alpha=a).fit(Xs[trn],Y[trn]).predict(Xs[tst])
        r2 = 1-((Y-Yh)**2).sum(0)/((Y-Y.mean(0))**2).sum(0)
        if r2.mean()>best: best=r2.mean(); bk=(a,np.round(r2,2))
    return bk
pr_ = RD2['prev_row'].values.astype(int)
for lab, rows, Y in [('G', rg, Yg[:,:2]), ('M', rm, Ym[:,:2])]:
    grp = RD2['stay_id'].values[rows]
    res_ = {}
    for nm, X in [('wc', WC0[rows]), ('hid', Hh0[rows]), ('wc+hid', np.hstack([WC0[rows],Hh0[rows]])), ('state', XALL[pr_[rows]])]:
        res_[nm] = cvr2(X, Y, grp)
    print(lab, res_)
```