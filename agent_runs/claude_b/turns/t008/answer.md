The insulin policy is clearly a sliding scale on the current glucose_finger reading:

| Reading (mg/dL) | Insulin given | Mean rate (u/h) |
|---|---|---|
| ≤ 140 | never | – |
| 140–160 | about 55% | about 1.0 |
| 160–180 | about 97% | about 1.1 |
| 180–220 | about 97% | about 2.1 |
| 220–250 | about 97% | about 3.1 |
| 250–300 | about 97% | about 3.9 |
| > 300 | about 99% | about 6.3 |

The previous rate barely matters once the reading is known, but there is a lot of spread within each cell (SD 0.4–2.0). That gives two possible sources of identifying variation. Sharp thresholds would allow a regression-discontinuity comparison just above and below a step, and the roughly 3% of hours with no insulin at high readings might be random omissions.

Next I will map the policy finely: rate and probability of insulin in 5 mg/dL steps of the reading, to locate the exact thresholds. I will also check what drives dosing when no finger reading is recorded that hour. Sharp jumps in the mean rate at particular readings would confirm a usable discontinuity design.

```python
Q['g5'] = (Q.glucose_finger//5*5).astype(int)
t = Q[(Q.g5>=130)&(Q.g5<=330)].groupby('g5').agg(n=('ins','size'), p_ins=('ins', lambda x:(x>0).mean()), mean_rate=('ins','mean'), rate_if=('ins', lambda x: x[x>0].mean()), sd_if=('ins', lambda x: x[x>0].std()))
print(t.round(3).to_string())
# when finger missing at hour t: what is dosing based on?
M = P[P.glucose_finger.isna() & (P.index.get_level_values(1)>=0)].copy()
print("hours with no finger reading:", len(M), " frac insulin given:", (M.ins>0).mean().round(3))
print("with lab reading:", M.glucose_lab.notna().mean().round(3))
```