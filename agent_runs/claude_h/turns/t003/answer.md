The records are hourly-ish event streams. There are 2.06M rows over 30k stays, so about 69 events per stay.
- Insulin rates run from 0.5 to 18.5 u/h (median 2).
- Norepinephrine rates run from 0.01 to 1.0 mcg/kg/min and are capped at 1.0.
- Norepinephrine dose = 4.8 × rate everywhere I looked, which implies a fixed 80 kg weight. Insulin dose = rate.
- The vocab gives a robust z = (x − median)/IQR, clipped to [−10, 10], and 16 fine bins per code.

Next I want the rest of the vocab (bin centres), the tensor list with shapes, and a sense of the stay structure. That means stay length, the reading cadence per hour, when insulin and norepinephrine run (from which hour, how continuously), and whether doses change within a stay. That structure decides which causal estimation strategy is possible. For example, if doses are set by a policy based on recent readings, I can adjust for those readings, as long as the policy has some randomness.

```python
print(json.dumps(V['value_bin_centres'])[:3000])
print(V['value_bin_note'])
TJ = json.load(open('/model/tensors.json'))
print(type(TJ), len(TJ))
for n in NAMES:
    print(n, SHAPES[n])
```