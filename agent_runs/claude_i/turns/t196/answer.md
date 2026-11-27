No candidate tracks the fitted column-0 value. A one-dimensional per-row fit just absorbs other head errors, so this search is not paying off. The local head is accurate for MAP (about 1 mg/dL), but for glucose it is off by about 3.6. I'll stop reverse-engineering it and rely on the tool's rollouts for calibration.

The bigger design question is where the dose deviation can be computed. At a drug bundle, z_post sees both u (the exact rate) and ctx (which should know the glucose reading taken 5 minutes earlier). A repurposed z_post hidden unit can compute rate − pibar(g) only if both parts are linearly readable. I'll test:
- pibar(g) and g from ctx at drug bundles;
- the insulin rate from u;
- the NE rate from u.

High R² for pibar from ctx (≥0.9) and for rate from u (≥0.95) would make a gated deviation unit in z_post feasible. Low R² would rule that design out.

```python
ctxD = ctx2[drugb].numpy(); UD = Uall[drugb]
gdec = np.array([gfh.get((s,j), glh.get((s,j), np.nan)) for s,j in zip(grp_,J_)])
pb_ = np.where(np.isnan(gdec), 0.0, pibar(np.nan_to_num(gdec, nan=100.0)))
insr_ = np.array([Rins.get((s,j),0.0) for s,j in zip(grp_,J_)]); ner_ = np.array([Rne.get((s,j),0.0) for s,j in zip(grp_,J_)])
def cvr2_(X_, y_, alpha=1.0):
    pr_ = np.zeros(len(y_))
    for trn,tst in gkf.split(X_,y_,grp_):
        pr_[tst] = Ridge(alpha=alpha).fit(X_[trn],y_[trn]).predict(X_[tst])
    return round(1-((y_-pr_)**2).sum()/((y_-y_.mean())**2).sum(),3), pr_
for nm_,X_ in [('ctx',ctxD),('u',UD),('ctx+u',np.c_[ctxD,UD])]:
    for tn_,y_ in [('pibar',pb_),('g',np.nan_to_num(gdec,nan=np.nanmean(gdec))),('insrate',insr_),('nerate',ner_)]:
        print(nm_, tn_, cvr2_(X_, y_)[0], end=' | ')
    print()
print('frac with reading', np.mean(~np.isnan(gdec)).round(3), 'ins present', np.mean(insr_>0).round(3))
```