The original model's answers on 100 tuning stays are:

| Edit | Shift | Stay-hours | Stays |
|---|---|---|---|
| Insulin ×2 | −1.85 mg/dL | 554 | 60 |
| Insulin ×½ | +0.78 mg/dL | 554 | 60 |
| Norepinephrine ×2 | −0.50 mmHg | 281 | 26 |
| Norepinephrine ×½ | −0.60 mmHg | 281 | 26 |

The insulin direction is right but looks far too small given my estimated response. The norepinephrine answers are near zero and the same sign for both edits, i.e. the model barely responds to NE. Only 26 of 100 stays count for NE (281 stay-hours) against 60 for insulin, so "reached" must have a specific definition.

Next I will inspect the per-stay-hour CSV:
- its columns;
- which hours count as "reached" (all hours after the first dose from hour 12, or only hours with a recorded reading);
- whether the NE count is limited to hours with an arterial reading.

Then I will compute what my estimated responses imply for exactly these stay-hours, giving the "true" target on this sample.

```python
import glob
f = '/work/fwd_1/dose_edit_original.csv'
de = pd.read_csv(f)
print(de.shape); print(de.columns.tolist()); print(de.head(20).to_string())
print(de.describe().T.to_string())
```