The scale is a step function of the 5-minute-earlier glucose_finger: none below about 150, then 1 u/h (150–180), 2 (180–220), 3 (220–260), 4 (260–300) and 6 (≥300). On top of that there is multiplicative noise with SD about 1/3 of the base, rounded to 0.5. This looks like a protocol plus a random multiplier. What matters is whether that multiplier is fresh each hour or a persistent per-patient factor, which could be a confounder such as insulin resistance.

Next:
- find the exact thresholds at 1 mg/dL resolution;
- compute the multiplier m = rate ÷ base;
- check its within-stay autocorrelation and how it relates to stay-level traits (mean glucose, arterial line, norepinephrine use).

High autocorrelation would point to a patient factor that needs careful handling. Near-zero autocorrelation would make each hour's multiplier a natural experiment.

```python
for lo,hi in [(145,156),(176,186),(216,226),(256,266),(296,306)]:
    s = g[(g.glucose_finger>=lo)&(g.glucose_finger<hi)]
    print(s.groupby('glucose_finger').agg(n=('ins','size'), p=('insulin', lambda x: x.notna().mean()), r=('insulin','mean')).round(3).T.to_string())
```