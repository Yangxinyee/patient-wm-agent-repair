The insulin rate looks like a sliding-scale base times multiplicative noise, rounded to 0.5. The base is 1 below 180, 2 for 180–220, 3 for 220–260 and about 4+ above. The coefficient of variation is about 0.33 in every band.

The deviations barely persist within a stay. The stay-mean variance share is 0.087, close to the 0.068 that independent noise with about 15 readings per stay would give. So the rate scatter behaves like independent hour-by-hour noise. That is a natural experiment, and the band jumps at 180/220/260 add regression-discontinuity variation. Between 140 and 160, starting is random with some hysteresis.

Next I'll compare finger and lab glucose at hours where both were taken, to learn how readings relate to an underlying glucose. Then I'll run a first estimate: regress the change in glucose_finger from hour t to t+1 on the insulin rate at t, with fine controls for the policy input (latest glucose) and the previous rate. A clear negative coefficient that is stable to the control set would point to a real insulin effect. A coefficient near zero once the policy inputs are controlled would mean the model's learned association may be pure confounding.

```python
both = np.isfinite(gf) & np.isfinite(gl)
d = gf[both] - gl[both]
print("paired finger-lab: n", both.sum(), "mean diff", d.mean().round(2), "sd", d.std().round(2), "corr", np.corrcoef(gf[both], gl[both])[0,1].round(3))
bm = np.isfinite(ma) & np.isfinite(mc)
dm = ma[bm]-mc[bm]
print("paired art-cuff: n", bm.sum(), "mean diff", dm.mean().round(2), "sd", dm.std().round(2), "corr", np.corrcoef(ma[bm], mc[bm])[0,1].round(3))
# consecutive finger readings: dynamics
rows=[]
for t in range(0,24):
    j=col(t)
    rows.append(pd.DataFrame({'t':t,'g0':gf[:,j],'g1':gf[:,j+1],'glast':glast[:,j],'ins':ins[:,j],'ins_prev':ins[:,j-1],
                              'ne':ne[:,j],'m':mlast[:,j],'stay':np.arange(len(ids))}))
R = pd.concat(rows, ignore_index=True)
Rv = R.dropna(subset=['g0','g1']).copy()
Rv['dg'] = Rv.g1-Rv.g0
import statsmodels.api as sm if False else None
```