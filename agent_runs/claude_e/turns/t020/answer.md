On these 100 tuning stays, the original model's answers are:

| edit | model's mean shift | stay-hours / stays |
|---|---|---|
| insulin x2 | −1.60 mg/dL | 818 / 93 |
| insulin x1/2 | +0.70 mg/dL | 818 / 93 |
| norepinephrine x2 | −0.15 mmHg | 693 / 62 |
| norepinephrine x1/2 | −0.40 mmHg | 693 / 62 |

So the model gives insulin only a weak lowering effect. From the records I expect roughly −10 to −20 mg/dL for x2. It gives norepinephrine essentially no effect; I haven't yet estimated the true norepinephrine effect.

To build a comparable estimate of the truth, I need exactly which stay-hours count as "reached". So next I'll inspect the per-stay-hour CSV: its columns, which hours are included, and how the shift varies with hours since hour 12. The time profile of the model's insulin shift shows whether the model has a delayed or persistent effect at all.

```python
import glob
f = '/work/fwd_1/dose_edit_original.csv'; print(glob.glob('/work/fwd_1/*'))
DE1 = pd.read_csv(f); print(DE1.shape); print(DE1.head(12).to_string()); print(DE1.columns.tolist())
for c in DE1.columns:
    if DE1[c].dtype==object or DE1[c].nunique()<30: print(c, DE1[c].value_counts().head(10).to_dict())
```