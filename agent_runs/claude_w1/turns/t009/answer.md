The rate jumps sharply at a recorded finger reading of 180 (about 1.05 to 2.08 u/h) and again at 220 (about 2.1 to 3.15). At exactly 180 or 220 the mean is halfway, which suggests the threshold applies to the unrounded reading.

So the clinicians' rule reads the recorded finger glucose: a step-function base rate times noise, rounded to 0.5. The sharp steps make a regression-discontinuity design possible. Readings at 179 and 181 come from essentially the same patients but get about +1 u/h more insulin, so comparing later glucose across a threshold estimates the causal effect even if the residual dose noise is confounded by something hidden.

Only 2.8% of insulin hours lack a current finger reading, and my lookup for those came back empty, so I need to see what drives them. Next I'll check that, and characterise the norepinephrine rule: rate against current arterial or cuff MAP, steps or a continuous rule, and on/off behaviour.

```python
print(nocur[['last_glucose_finger','glucose_lab','last_glucose_lab','insulin','ins_prev']].describe().to_string())
print(nocur.head(10).to_string())
# NE policy
neever = P.groupby('stay_id').ne_on.transform('max')==1
N = P[neever & (P.h>=0)].copy()
N['map_now'] = N.map_arterial.fillna(N.map_cuff)
N['last_map'] = N.last_map_arterial.where(N.last_map_arterial.notna(), N.last_map_cuff)
N['mbin'] = pd.cut(N.last_map_arterial.fillna(N.last_map_cuff), [0,40,50,55,60,65,70,75,80,90,200])
print(N.groupby(['ne_prev_on','mbin']).ne_on.agg(['mean','size']).to_string())
print(N[N.ne_on==1].groupby(['mbin']).norepinephrine.agg(['mean','std','size']).to_string())
```