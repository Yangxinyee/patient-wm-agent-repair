The original model's readouts are already close to 0:

| Edit | Mean shift |
|---|---|
| insulin x2 | +0.114 mg/dL |
| insulin x1/2 | −0.148 mg/dL |
| norepinephrine x2 | +0.005 mmHg |
| norepinephrine x1/2 | −0.006 mmHg |

That agrees with my estimate that both drugs are inert. It is also suspicious, because the task suggests the model answers dose questions badly. So I must make sure I am not missing a real effect that the records reveal some other way.

One possibility I have not excluded: the recorded multiplier could be recording noise rather than the dose actually given. Then the margin analysis would be blind to a real effect of the base dose. The 150 regression discontinuity already argues against this, but the other thresholds (180, 220, 260, 300 for insulin; the titration break at MAP 75 for continuing norepinephrine) test it directly.

Next I will:
1. Check the per-stay-hour CSV, to see whether the model's tiny mean hides large shifts that cancel.
2. Run regression discontinuities at every insulin threshold.

If there are no jumps anywhere and the model's per-hour shifts are small, "inert" stands and the model is roughly right already.

```python
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de.columns.tolist()); print(de.head(10).to_string())
print(de.describe().round(3).to_string())
```