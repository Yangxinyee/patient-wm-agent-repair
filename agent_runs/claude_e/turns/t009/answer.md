The protocol thresholds are exact: 150, 180, 220 and roughly 300 mg/dL. At exactly the threshold value about half the records fall on each side, consistent with the clinician seeing the reading before rounding. Within each tier the rate looks like base (1, 2 or 3 u/h) times a random factor, rounded to 0.5. For example, the 150–180 tier gives 0.5/1.0/1.5/2.0 in proportions 22k/79k/26k/3.6k.

The important question is whether that random factor is independent of the patient. If it is a per-stay factor (a clinician style, or a hidden patient trait such as insulin resistance), within-tier comparisons would be confounded. Next I'll check whether the within-tier multiplier persists within a stay (autocorrelation) and whether it relates to anything recorded (hour, previous glucose, glucose_lab, MAP, norepinephrine).

```python
base_map = {'150-180':1.0,'180-220':2.0,'220-300':3.0}
Q = Pn[Pn.tier.isin(base_map.keys()) & Pn.on_ins & ((Pn.gf_now%10)!=0)].copy()   # avoid exact-threshold ties
Q['base'] = Q.tier.map(base_map).astype(float); Q['mult'] = Q.ins/Q.base
Q = Q.sort_values(['stay','h'])
Q['mult_prev'] = Q.groupby('stay').mult.shift(1); Q['h_prev'] = Q.groupby('stay').h.shift(1)
q = Q[Q.h_prev==Q.h-1]
print('corr(mult, mult_prev) consecutive hours:', np.corrcoef(np.log(q.mult), np.log(q.mult_prev))[0,1], len(q))
# stay-level share of variance
lm = np.log(Q.mult); sm = lm.groupby(Q.stay).transform('mean'); cnt = Q.groupby('stay').mult.transform('size')
print('var log mult', lm.var(), ' within-stay var', (lm-sm)[cnt>=5].var())
for c in ['h','gl_last','ma_last','ne','ins_prev','gf_last']:
    pass
Q['gf_prev'] = Q.groupby('stay').gf_now.shift(1)
import statsmodels.api as sm_ if False else None
from numpy.linalg import lstsq
for c in ['h','gl_last','ma_last','gf_prev']:
    d = Q[[c,'mult','gf_now']].dropna(); print(c, 'corr with log mult:', round(np.corrcoef(d[c], np.log(d.mult))[0,1],4), len(d))
print('NE given (ne notna) vs log mult:', Q.groupby(Q.ne.notna()).mult.apply(lambda x: np.log(x).mean()).to_string())
```