## Step 24: norepinephrine noise is i.i.d.; MAP measurement is state-independent

**What the last result showed.**
- The norepinephrine step noise has no autocorrelation (r = −0.003), so it is per-record noise.
- MAP measurement never depends on MAP or on norepinephrine:
  - stays with an arterial line (fixed at hours 0–1, about 14.5k stays) have arterial MAP in 95% of hours and cuff in 30%;
  - other stays have cuff in 90% of hours.
- So both MAP readings are selection-free outcomes, given the stay type, which is set before treatment.

**What I will do next.** Build valid norepinephrine surprises, e = rate − E[rate | observed history]. The expected rate must capture every predictable part of the policy, or the "surprise" will be correlated with MAP. So I need to check:
- whether the drift depends on the actual MAP inside a band, not just the band;
- how it depends on the previous rate (the cap at 1.0, the floor at 0.01, rounding).

I will tabulate the mean log ratio by finer MAP bins for previous rates below 0.4, where the cap rarely binds.

**What would count as evidence.**
- A flat drift inside each band means the band-and-previous-rate cells fully describe the policy.
- A drift that varies with MAP inside a band means the expected rate needs MAP as a continuous input.

```python
far = on[(on.ne_prev<0.4)&(on.ne_rate>0)].copy()
print(far.groupby(pd.cut(far.lm_art_pref,[0,40,45,50,55,58,60,62,63,64,65,66,70,73,75,76,78,80,90,200])).lr.agg(['mean','std','size']).round(3).to_string())
# dependence on previous rate within band (far from cap)
far['b3'] = np.where(far.lm_art_pref<65,0,np.where(far.lm_art_pref<=75,1,2))
print(far.groupby(['b3', pd.cut(far.ne_prev,[0,0.02,0.05,0.1,0.2,0.3,0.4])]).lr.agg(['mean','std','size']).round(3).to_string())
```