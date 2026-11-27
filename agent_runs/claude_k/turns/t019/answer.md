The insulin shocks pass every balance check. They are unrelated to:
- the same-hour lab glucose the policy didn't use (0.0001 ± 0.0003),
- glucose at t-1 and t-2,
- the hour.

That supports exogenous scatter, so the insulin estimates should be unbiased.

The norepinephrine shocks are not clean as I defined them. They correlate strongly with MAP at t-1 (t ≈ 27) and the previous rate, and slightly with the same-hour cuff reading. So the titration rule uses more inputs than my (previous rate × MAP band) model. My MAP response estimates may be biased through the omitted MAP history.

Next I'll pin down the norepinephrine policy. Which MAP reading drives the 65/75 thresholds when arterial and cuff are both present? Does the step depend on the previous MAP (a trend or average)? How do the previous rate and the 1.0 cap enter? A sharp discontinuity in one specific variable (e.g. arterial MAP, or an average of current and previous) would identify the policy input. I can then rebuild the shock as a true residual and redo the response estimates.

```python
# which MAP drives threshold at 65 and 75? use rows on norepi with prev rate in (0.05,0.9)
Q = []
for t in range(1,25):
    j=col(t)
    Q.append(pd.DataFrame({'t':t,'ma':ma[:,j],'mc':mc[:,j],'mal':mlast[:,j],'mm1':mlast[:,j-1],'r':ne[:,j],'rp':ne[:,j-1]}))
Q = pd.concat(Q, ignore_index=True)
Q = Q[(Q.rp>0.05)&(Q.rp<0.9)&(Q.r>0)].copy(); Q['lr']=np.log(Q.r/Q.rp)
bothq = Q.dropna(subset=['ma','mc'])
print("both present, lr by arterial (rows) at fixed cuff ranges:")
print(bothq.assign(A=bothq.ma.clip(60,80), C=pd.cut(bothq.mc,[0,62,68,72,78,200])).pivot_table(index='A', columns='C', values='lr', aggfunc='mean').round(2))
onlyc = Q[Q.ma.isna() & Q.mc.notna()]
print("cuff only, lr by cuff:", onlyc.groupby(onlyc.mc.clip(60,80)).lr.mean().round(2).to_dict())
none = Q[Q.ma.isna() & Q.mc.isna()]
print("no MAP at t: n", len(none), " lr by mlast:", none.groupby(none.mal.clip(60,80)).lr.mean().round(2).to_dict())
```