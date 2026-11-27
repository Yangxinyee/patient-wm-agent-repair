The within-code covariance of `with_code` is nearly degenerate: several zero eigenvalues and a median of 1e-4. Most of the state variation lies in a few directions. The Mahalanobis distances between glucose and MAP codes are therefore enormous (200–300). Part of that comes from directions with almost no variance on recorded data, which may not hold up on counterfactual rollout states.

So I will build a deliberately shrunk gate. I take the LDA direction between {glucose_finger, glucose_lab} and {map_arterial, map_cuff} with ridge λ ∈ {1e-3, 1e-2, 1e-1}, scaled so the class means differ by 20. Then I check the per-code min/max of the projection and the weight norm.

What I am looking for is a moderate λ whose ranges are clearly separated: every glucose sample above the midpoint by more than 8, every MAP sample below by more than 8. That would give a robust gate.

```python
XT=np.concatenate([Xs_['glucose_finger'],Xs_['glucose_lab']]); XN=np.concatenate([Xs_['map_arterial'],Xs_['map_cuff']])
mT,mN=XT.mean(0),XN.mean(0); mid=(mT+mN)/2
for lam in [1e-3,1e-2,1e-1,1.0]:
    wg=np.linalg.solve(Sw+lam*np.eye(192),mT-mN); wg=wg*20/((mT-mN)@wg)
    r={c:((Xs_[c]-mid)@wg) for c in MEAS}
    print(lam,'|w|',round(float(np.linalg.norm(wg)),2),{c[:6]+c[-3:]:(round(float(r[c].min()),1),round(float(r[c].max()),1),round(float(r[c].std()),2)) for c in MEAS})
```