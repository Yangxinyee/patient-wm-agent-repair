The saved Jacobians are confirmed: Jtok holds the effect of a z-token change on the next predictions [finger glucose, lab glucose, arterial MAP, cuff MAP] at 128 bundles, half insulin and half norepinephrine. The other building blocks are also in place:
- There are 58 dead z-posterior units with near-zero activation, so repurposing them is free.
- Insulin rates average 1.86, correlate 0.80 with the policy-expected rate, and deviate from it with sd 0.67.

Before building the injection, I need to know whether it would help or hurt factual fit. An injected effect of the dose deviation improves the factual likelihood only if the original model's residuals still contain that effect. For insulin, I'll regress glucose residuals (observed − model mean, from the saved factual output) on the insulin deviation (rate − pibar) at lags 1–3 h. For MAP, I'll regress residuals on NE rates at lags 1–3.

The decision rule:
- A glucose coefficient near the kernel's −2 per u/h means the model misses the deviation effect, so a calibrated injection is factual-safe. A coefficient near 0 would refute that.
- For NE, the pattern of lag coefficients tells me whether a rate-level or rate-change injection is factual-safe.

```python
Fd=F2.copy(); Fd['res']=Fd.observed-Fd.predicted_mean
print('F2 stays in S3:', np.isin(Fd.stay_id.unique(), np.unique(S3a)).mean(), 'events', Fd.event.value_counts().to_dict())
idx={(S3a[k],round(H3a[k],2)):k for k in range(5156)}
def at(s,h,arr,flag):
    k=idx.get((s,round(h+0.083,2))); return float(arr[k]) if (k is not None and flag[k] and np.isfinite(arr[k])) else 0.0
devA=np.where(mI, rI-pb, 0.0)
for evn,arr,flag in [('glucose_finger',devA,hi_),('glucose_lab',devA,hi_),('map_arterial',rN,hn_),('map_cuff',rN,hn_)]:
    t=Fd[Fd.event==evn]; H=np.round(t.hour.values)
    Xr=np.array([[at(s,h-l,arr,flag) for l in (1,2,3)] for s,h in zip(t.stay_id.values,H)])
    A_=np.c_[np.ones(len(t)),Xr]; y_=t.res.values; cf,_,_,_=np.linalg.lstsq(A_,y_,rcond=None)
    rs=y_-A_@cf; se=np.sqrt(np.diag(np.linalg.inv(A_.T@A_))*rs.var())
    print(evn,len(t),'coef',cf[1:].round(2),'se',se[1:].round(2),'res sd',y_.std().round(2),'R2',(1-rs.var()/y_.var()).round(4))
```