The README confirms that doses were chosen by clinicians. That is the classic confounding-by-indication setup: insulin is given when glucose is high, so a naive model may learn that more insulin goes with higher glucose. The vocabulary gives the normalisation:
- insulin rate: median 1.5, IQR 1.0;
- norepinephrine rate: median 0.46, IQR 0.6;
- glucose_finger bin centres span 72 to 271 mg/dL.

One limitation matters for later: the predicted mean is a weighted sum of fixed bin centres, so predicted shifts are bounded by that bin structure.

Next I'll print the full numeric table (the output was cut off) and all 181 tensor names with shapes. Then I'll load train.csv to see the event structure: how often each event happens, the dose distributions, and when infusions run.

```python
for n in V['numeric']:
    print(n)
print()
for nm in NAMES:
    print(nm, SHAPES[nm])
```