The insulin policy is a clean sliding scale on the finger reading at that hour. The tiers start at 150, 180, 220, 260 and 300 mg/dL, with typical rates of about 1.05, 2.1, 3.13, 4.19 and 6.2 u/h. Within a tier the rate varies with a constant coefficient of variation of about 33% (multiplicative noise, rounded to 0.5). About 3% of doses are omitted. Insulin is almost never started in an hour without a finger reading (only 1.9% of such hours).

This gives two identification strategies:

- **Regression discontinuity.** At each threshold the dose jumps by about 1 u/h while the patient's true glucose varies smoothly across it.
- **Regression adjustment.** If the multiplicative noise is independent of the patient's hidden state, then given the reading, dose variation is as good as random.

If the two strategies agree, the noise is benign. If they disagree, the discontinuity estimate is the more credible one.

Next I will characterise the norepinephrine policy against MAP the same way, in 2 mmHg steps, separately for arterial and cuff readings. That tells me which reading drives the decision, where the thresholds are and how noisy the rate is.

```python
# NE policy: which MAP reading drives it? Use arterial if present else cuff
P['map_any'] = P.map_arterial.fillna(P.map_cuff)
for col in ['map_arterial','map_cuff']:
    R = P[P[col].notna()].copy()
    R['m2'] = (R[col]//2*2).astype(int)
    t = R[(R.m2>=40)&(R.m2<=96)].groupby('m2').agg(n=('ne','size'), p_ne=('ne', lambda x:(x>0).mean()), mean_rate=('ne','mean'), rate_if=('ne', lambda x: x[x>0].mean()), sd_if=('ne', lambda x: x[x>0].std()), prev=('ne_prev','mean'))
    print(col); print(t.round(3).to_string())
nomap = P[P.map_arterial.isna() & P.map_cuff.isna() & (P.index.get_level_values(1)>=0)]
print("hours no MAP:", len(nomap), " frac NE:", (nomap.ne>0).mean().round(3))
```