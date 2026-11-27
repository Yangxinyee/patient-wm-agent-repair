The insulin rule has a sharp threshold. At finger glucose ≥ 150 insulin is always given, below about 145 never, and in the 145–150 band only 10% of the time (probably readings of exactly 150).

Above the threshold the rate spreads widely at a fixed glucose. At 190–199 mg/dL it runs from 1.0 to 4.5 U/h with a mode of 2, and the spread grows with glucose, which looks multiplicative.

That gives me two possible natural experiments:
1. **Threshold comparison (regression discontinuity).** Finger sticks are probably noisy readings of true glucose, so landing just above or just below 150 is close to a coin flip. Comparing later glucose for readings just below vs just above 150 estimates the effect of starting insulin.
2. **Leftover rate variation at a given glucose.** This works only if that variation is not tied to the patient's hidden state.

Next I will pin down the exact threshold, check what triggers the 2% of insulin records given without a finger reading, and model the rate.
- Fit log(rate) on log(glucose − 150) or similar.
- Measure how much of the residual is persistent within a stay (stay-level scale) versus fresh each hour.
- Test whether it depends on the previous rate or the glucose trend.

```python
# exact threshold
xg = x[x.glucose_finger.notna() & (x.h>=0)]
for g in range(144,156):
    s = xg[xg.glucose_finger==g]
    print(g, len(s), s.insulin.notna().mean().round(3))
# insulin without finger glucose: what's present?
ni = x[x.insulin.notna() & x.glucose_finger.isna()]
print(len(ni), ni[['glucose_lab','glucose_finger_prev','insulin_prev']].describe().to_string())
print(ni.head(10).to_string())
```