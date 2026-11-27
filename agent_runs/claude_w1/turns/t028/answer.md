The original model barely responds to dose changes:

| Edit | Model's predicted shift |
|---|---|
| Insulin ×2 | −1.45 mg/dL |
| Insulin ×1/2 | +0.82 mg/dL |
| Norepinephrine ×2 | +0.15 mmHg |
| Norepinephrine ×1/2 | −0.41 mmHg |

My estimates from the records are roughly −15 mg/dL for insulin ×2 and +3 to +4 mmHg for norepinephrine ×2. So the model's drug effects are about 10× too weak for insulin and nearly absent for norepinephrine. That fits a model that put the credit on the readings that drive the doses rather than on the doses.

The readout covers only certain stay-hours: 763 hours over 89 insulin stays (about 8.6 per stay) and 518 hours over just 47 norepinephrine stays. Next I'll open the per-stay-hour file to see which rows count as "reached". Probably only hours with an actual reading, or arterial-line stays. I need that to compute matching "true" shifts on exactly these rows.

```python
import glob
print(os.listdir('/work/fwd_1'))
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de.shape); print(de.columns.tolist())
print(de.head(30).to_string())
print(de.describe(include='all').T.to_string())
```