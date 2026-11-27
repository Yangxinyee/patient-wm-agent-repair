Insulin's immediate effect is about -2 mg/dL per unit/h in every glucose band: -1.8 to -2.0 at typical rates of 1–3 units/h, and -2.25 near 4.3 units/h. Within bands, the mean next-hour change falls roughly linearly with the actual rate, about 2 mg/dL per unit from 0.5 up to 7 units/h. So the dose-response looks linear and additive, not dependent on the glucose level. The x0/x1.5/x2/x3 effects should then scale with (multiplier - 1) × the recorded doses, filtered through the persistence profile.

Next I'll do the same for norepinephrine:
- its dosing policy (the rate as a function of the latest MAP and the previous rate),
- its 0.01-step scatter and the cap at 1.0.

A policy with independent scatter like insulin's would let me estimate the MAP impulse response the same way. The cap at 1.0 truncates the noise at the top, so I'll keep capped records out of the deviation analysis.

```python
D['mbin'] = pd.cut(D.m, [0,50,55,60,62,64,65,66,68,70,75,80,90,200])
print(D.groupby('mbin').agg(n=('ne','size'), p_on=('ne', lambda x:(x>0).mean()), mean_rate_on=('ne', lambda x: x[x>0].mean()), sd_on=('ne', lambda x: x[x>0].std())).round(3))
print(pd.crosstab(pd.cut(D.ne_prev,[-.01,0,.1,.2,.4,.6,.8,.99,1.0]), pd.cut(D.ne,[-.01,0,.1,.2,.4,.6,.8,.99,1.0]), normalize='index').round(2))
```