The norepinephrine estimate is not robust to the choice of controls, so I can no longer treat it as clean evidence:

| Control set | Implied ×2 readout (se) | Lag-1 coefficient |
|---|---|---|
| Full (as before) | −0.40 (0.13) | +0.03 |
| Minimal (current MAP, previous rate, hour) | −1.05 (0.15) | −0.39 |
| Extended | −1.37 (0.15) | — |

Two of these runs are invalid because of bugs I found:
- The extended set's cumulative-mean-rate feature includes the current rate, so it leaks the treatment and that run is void.
- My "hours into course" feature is all zeros, because a NaN propagated through the cumulative sum. That is also why the course-length split crashed.

The minimal-versus-full gap is real and informative. If the titration rule depended only on current MAP and the previous rate, adding more history could not move the estimate this much. So the rule must also depend on earlier rates or MAPs, which also predict future MAP. My "innovation" then still contains part of the systematic titration, and the norepinephrine estimate is sensitive to how fully I model the rule.

The insulin rule (sliding scale × jitter, rounded) is fully captured, so it should not have this problem.

Next I will:
1. Fix both features (cumulative rate up to t−1 only; hours into course computed properly).
2. Measure how much each block of history (rates at t−2 and t−3, MAP at t−1 and t−2, longer summaries) adds to predicting the norepinephrine rate. This tells me what the titration rule really uses.
3. Re-estimate with a correctly built extended set, to see whether the estimate settles as controls are added.
4. Run the same minimal-versus-full test for insulin.

- **Supports a clean instrument for insulin:** its estimate is unchanged between control sets.
- **Supports trusting the full norepinephrine estimate:** that estimate stabilises once the policy's real inputs are included.

```python
t0=time.time()
NEpos=np.where(NE0>0,NE0,np.nan)
def cummean_prev(X):  # mean over columns < t
    m=np.where(np.isnan(X),0,X); c=(~np.isnan(X)).astype(float)
    M=np.zeros_like(m); C=np.zeros_like(c); M[:,1:]=m.cumsum(1)[:,:-1]; C[:,1:]=c.cumsum(1)[:,:-1]
    with np.errstate(invalid='ignore',divide='ignore'): return np.where(C>0,M/np.maximum(C,1),np.nan)
onprev=np.zeros_like(NE0); onprev[:,1:]=(NE0[:,:-1]>0).astype(float).cumsum(1)
feats['non']=onprev
# policy diagnostics: cross-fitted residual variance of r_t on nested feature sets
def pol_resvar(fset):
    X_=np.stack([fset[n][si,ti] for n in fset],1); y=NE[si,ti]; fold_=((stay_ids[si]//7)%2)
    pred=np.zeros(len(y))
    for f in (0,1):
        tr=fold_!=f; te=fold_==f
        pred[te]=HGB(max_iter=300,max_leaf_nodes=31,min_samples_leaf=50).fit(X_[tr],y[tr]).predict(X_[te])
    return np.var(y-pred)
base_set={'ga0':GA,'gc0':GC,'r1':lag(NE0,1)}
sets=[('MAP_t,r1',base_set),
      ('+hr',{**base_set,'hr':hourmat}),
      ('+r2,r3',{**base_set,'r2':lag(NE0,2),'r3':lag(NE0,3)}),
      ('+ga1,gc1,ga2,gc2',{**base_set,'ga1':lag(GA,1),'gc1':lag(GC,1),'ga2':lag(GA,2),'gc2':lag(GC,2)}),
      ('+course len',{**base_set,'non':onprev}),
      ('+cum means',{**base_set,'cmA':cummean(GA),'cmC':cummean(GC),'cmin':feats['cmin']}),
      ('+cmr_prev',{**base_set,'cmr':cummean_prev(NEpos)})]
for nm,fs in sets: print('%-18s resid var %.5f'%(nm,pol_resvar(fs)))
print('time',round(time.time()-t0,1))
```