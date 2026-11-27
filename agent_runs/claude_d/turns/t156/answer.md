The decoders show where the information sits. From the value-head input, the raw rates decode well: lag-1 insulin rate R² 0.83, lag-1 NE rate 0.78. The surprises decode poorly: insulin lag-1 0.16, lag-2 0.02. That is expected, since the surprise is the rate minus a step-function expectation, and a linear readout sees the rate error and the expectation error together.

So the correction should be built from two decoders: a well-decoded rate minus a decoded expectation. Its factual cost will then come from decoding noise, scaled by the square of the gain.

Before committing, I want to compare like with like against the notebook's earlier R² of 0.5–0.6. I'll restrict to glucose readings that directly follow a real insulin record, and compare decoding from `with_code`, from `out`, and from the full state. If `with_code` is much worse than the full state, information is being lost before the head and I should think about where else to read it.

```python
gi = gmask & (RD2['ins_1'].fillna(0).values>0)
Xout = XALL[RD2['prev_row'].values.astype(int)][:, :192]; Xfull = XALL[RD2['prev_row'].values.astype(int)][:, :960]
def r2_from(X_, mask, y, alphas=(0.1,1,10,100)):
    Xz = (X_[mask]-X_[mask].mean(0))/(X_[mask].std(0)+1e-6)
    pr = cross_val_predict(RidgeCV(alphas=alphas), Xz, y, cv=KFold(5,shuffle=True,random_state=0))
    return round(1-((y-pr)**2).sum()/((y-y.mean())**2).sum(),3)
d = RD2[gi]; y1=d.eps_ins_1.values; yr=d.ins_1.values; yb=d.ins_exp_1.values
print('n',gi.sum(),'sd eps',y1.std().round(3),'sd ins',yr.std().round(3))
for nm,X_ in [('wc',WC0),('out',Xout),('full',Xfull)]:
    print(nm, 'eps1', r2_from(X_,gi,y1), 'ins1', r2_from(X_,gi,yr), 'exp1', r2_from(X_,gi,yb))
```