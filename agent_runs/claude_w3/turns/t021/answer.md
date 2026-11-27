The original model predicts almost no drug effects:

| edit | model's mean shift |
|---|---|
| insulin ×2 | −0.39 mg/dL |
| insulin ×½ | +0.15 mg/dL |
| norepinephrine ×2 | −0.66 mmHg |
| norepinephrine ×½ | −0.24 mmHg |

Both norepinephrine edits go the same direction. My provisional insulin truth is about −14 for ×2, so the model largely ignores insulin's causal effect.

The norepinephrine readout covered only 58 of the 90 stays I selected with norepinephrine from hour 12. That suggests it counts only stay-hours with a map_arterial reading, i.e. stays with an arterial line. The insulin readout used 815 stay-hours in 90 stays.

Next I will inspect the per-stay-hour CSV to pin down which rows count: whether a recorded outcome reading is required at that hour, and what "reached" means. Then I will compute my provisional true effect on exactly those rows. If the rows are hours with a recorded reading after the first dose at hour ≥ 12, my calculation must be restricted the same way.

```python
import glob
f = '/work/fwd_1/dose_edit_original.csv'
de = pd.read_csv(f)
print(de.shape); print(de.columns.tolist()); print(de.head(30).to_string())
print(de.describe(include='all').T.to_string())
```