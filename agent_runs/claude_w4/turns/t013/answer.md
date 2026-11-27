## Step 12: first regression gives about −2 mg/dL per unit/h, but the placebos are not clean

**What the last result showed.**
- With controls, the next-hour finger glucose falls by about 2.0 mg/dL per unit/h of insulin at hour t (SE 0.05).
- However, the previous hours' insulin rates enter with large positive coefficients (+2.2, +2.6). That is typical of confounding by past glucose levels: the current noisy reading underestimates how high the hidden glucose has been.
- The placebo correlations are small but clearly non-zero: corr(noise, recent glucose change) = −0.038, and corr(noise, previous dose) = −0.069.
- The spline design was partly collinear, so some SEs are meaningless.

**What this means.** The within-band noise is not obviously pure randomness, so I should not trust this estimate yet.

**What I will do next.** Look at the measurement model:
- How do finger and lab readings at the same hour relate (offset, noise)? This tells me what hidden glucose looks like.
- How noisy are readings, judged by their autocorrelation?

That helps judge whether the clinician could be reacting to hidden glucose, or to the reading only. I will also redo the regression with richer history controls (previous readings) and see whether the insulin coefficient is stable.

**What would count as evidence.**
- A stable coefficient across control sets, with placebo correlations that vanish once history is controlled, would support the noise being exogenous.
- Instability would push me towards the threshold-discontinuity design.

```python
both = x[x.glucose_finger.notna() & x.glucose_lab.notna()]
d = both.glucose_finger - both.glucose_lab
print("finger-lab same hour: mean", d.mean().round(2), " sd", d.std().round(2), " corr", np.corrcoef(both.glucose_finger, both.glucose_lab)[0,1].round(3), " n", len(both))
# regress finger on lab
print("slope finger~lab:", np.polyfit(both.glucose_lab, both.glucose_finger, 1).round(3))
bm = x[x.map_arterial.notna() & x.map_cuff.notna()]
dm = bm.map_arterial - bm.map_cuff
print("arterial - cuff same hour: mean", dm.mean().round(2), " sd", dm.std().round(2), " corr", np.corrcoef(bm.map_arterial, bm.map_cuff)[0,1].round(3), " slope art~cuff", np.polyfit(bm.map_cuff, bm.map_arterial,1).round(3), " n", len(bm))
# richer regression for insulin effect
x['gf_prev2'] = x.groupby('stay_id').glucose_finger.shift(2)
s = x[(x.age_glucose_finger==0) & (x.ins_rate>0) & x.gf_next.notna() & x.gf_prev.notna() & x.gf_prev2.notna()].copy()
gf = s.glucose_finger.values
spl = [np.maximum(gf-k,0) for k in [165,180,200,220,240,260]]
bd = [(s.band.values==k).astype(float) for k in [2,3,4]]
base_X = [gf]+spl+bd
names = ['gf']+[f'sp{k}' for k in range(6)]+['b2','b3','b4']
for extra, en in [([],[]),
                  ([s.gf_prev.values, s.gf_prev2.values],['gfp1','gfp2']),
                  ([s.gf_prev.values, s.gf_prev2.values, s.prev_ins_rate.values, s.ins_prev2.values],['gfp1','gfp2','insp1','insp2'])]:
    r = ols(s.gf_next.values, [s.ins_rate.values]+base_X+extra, ['ins']+names+en)
    print(r.loc[['ins']+en].round(3).T.to_string())
print("n", len(s))
print("corr(noise, gfp1 resid):", np.corrcoef(s.noise, s.gf_prev - s.glucose_finger)[0,1])
```