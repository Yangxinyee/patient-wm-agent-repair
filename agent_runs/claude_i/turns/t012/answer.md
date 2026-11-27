Insulin lowers glucose in this ICU, and the effect looks linear in rate. The within-band random-rate slope is about −1.9 to −2.0 mg/dL per u/h at the next hour, very consistent across all three bands and on both finger and lab readings. The regression discontinuity at 150 gives a −1.40 ± 0.18 drop for a rate of about 1.05. That is somewhat smaller than linear extrapolation through zero would give (≈ −2.0).

The gap could have three explanations: real nonlinearity, a lag (the drug runs only 55 of the 60 minutes before the next reading), or selection in which hours get a next reading. The last matters because insulin probably triggers hourly finger-sticks.

Next I'll check:
- the next-hour effect by individual rate level within a band, for the shape of the dose-response;
- whether the probability of a next-hour reading depends on insulin or on glucose, for selection.

```python
Q['gf_next_present'] = Q.gf_next.notna()
Q['any_next_present'] = Q.gf_next.notna() | Q.gl_next.notna()
s = Q[(Q.gdec>=135)&(Q.gdec<165)]
print("P(gf next present) by gdec:", s.groupby(s.gdec.round(-0) // 5 * 5).gf_next_present.mean().round(3).to_dict())
print("P(gf next present | ins given) vs not, overall:", Q.groupby(Q['ins'].notna()).gf_next_present.mean().round(3).to_dict())
# dummies by rate within band 150-180
for lo,hi,rates in [(150,180,[0.5,1.0,1.5,2.0]),(180,220,[1.0,1.5,2.0,2.5,3.0,3.5,4.0])]:
    s = Q[(Q.gdec>=lo)&(Q.gdec<hi)&Q['ins'].isin(rates)]
    X = [s.gdec] + [(s['ins']==r).astype(float) for r in rates[1:]]
    res, n, sd = ols(s.gf_next, X, ['gdec']+[f'r{r}' for r in rates[1:]])
    print(f"band {lo}-{hi} effects relative to rate {rates[0]}:", res.b.round(2).to_dict(), "se", res.se.round(2).to_dict())
# effect at lag 2..6 of random rate (within band 150-180), naive (total effect incl. feedback)
for k in range(1,7):
    Q[f'gf_lead{k}'] = P.groupby(level='stay_id').gf.shift(-k).reindex(Q.index)
s = Q[(Q.gdec>=150)&(Q.gdec<180)&Q['ins'].notna()]
print("total effect of rate at h on gf at h+k (band 150-180):", {k: tuple(ols(s[f'gf_lead{k}'], [s.gdec, s['ins']], ['g','rate'])[0].loc['rate'].round(2)) for k in range(1,7)})
```