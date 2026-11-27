The calibration gives a clear picture. The table shows predicted / true shift in glucose_finger (mg/dL) on 100 training stays:

| Model | x2 | x1/2 | x0 | x2 from 16 |
|---|---|---|---|---|
| Unedited | -3.7 / -16.5 | +2.2 / +8.3 | +2.5 / +16.5 | -2.6 / -14.2 |
| Oracle, kap = 0 | -53 / -16.5 | | | |
| Oracle, kap = 0.42 | -26 / -16.5 | | | |
| Oracle, kap = 1 | -13.7 / -16.5 | +7.0 / +8.3 | +11.7 / +16.5 | -11.3 / -14.2 |

- **The model underestimates the insulin effect about 4.5-fold.** Its shift grows roughly in proportion to the truth over the lags.
- **The model carries a shifted reading forward almost completely.** So the direct shift it needs is mostly a lag-1 deviation term (G ≈ -2.0, -0.5, then small positive terms), not the full true kernel.
- **Kap = 1 is close but undershoots.** It is too large at lag 1 (-4.5 vs -3.6) and too small later. A kap around 0.85, with weight moved from lag 1 to lags 2–3, should fit.
- **Shifting the head's predicted mean by a deviation-based term works when the shift is computed exactly.** Getting the right kernel is a calibration step.

The open question is whether the weights can implement it. The value head only sees the model state, so the state at a glucose reading must linearly encode the recent insulin dose and its policy dose (lags 1–2 above all). I'll measure that next. First I need the head code and the head tensor names, to see where a code-gated linear readout can go. Candidates:
- a rank-1 addition to the value MLP's output layer, reading a combination of its 192 hidden units;
- a repurposed hidden unit, gated by the code embedding.

This run prints head.py, the head and embedding tensor names with shapes, and the bin centres of the glucose and MAP codes.

```python
print(open('/work/head.py').read())
print([ (n,SHAPES[n]) for n in NAMES if ('head' in n or 'attr' in n or 'query' in n or 'type' in n or 'transition' in n or 'z_head' in n or 'z_prior' in n)])
for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    print(c, CODE[c], np.round(CENT[CODE[c]],1))
```