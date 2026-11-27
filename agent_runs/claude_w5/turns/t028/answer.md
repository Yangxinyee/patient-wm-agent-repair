The original model predicts almost no drug effects:

| readout | model (mean shift) |
|---|---|
| insulin x2 | −0.55 mg/dL |
| insulin x1/2 | +0.003 mg/dL |
| norepinephrine x2 | +0.19 mmHg |
| norepinephrine x1/2 | −0.17 mmHg |

The records say insulin and norepinephrine have substantial causal effects, so the model's dose response is far too weak. It probably relies on the readings and treats the dose atoms mostly as signals of patient state.

The averages are over "reached" stay-hours: 703 hours across 73 stays for insulin and 822 hours across 74 stays for norepinephrine.

Next I'll open the per-stay-hour CSV to learn:
- exactly which hours count as reached (the first edited dose before the hour, or any reading hour);
- whether the mean is pooled over stay-hours.

Then I'll compute my g-estimation-based true shift for exactly those stay-hours. That gives the target the edited model must hit on these stays.

```python
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de.shape); print(de.columns.tolist()); print(de.head(30).to_string())
print(de.describe().T.to_string())
```