## Step 32: the original model barely responds to doses

**What the last result showed.** The original model's readouts on the 100 tuning stays are all almost zero:

| edit | readout | stay-hours | stays |
|---|---|---|---|
| insulin ×2 | −0.35 mg/dL | 919 | 99 |
| insulin ×1/2 | +0.21 mg/dL | 919 | 99 |
| norepinephrine ×2 | +0.02 mmHg | 730 | 67 |
| norepinephrine ×1/2 | −0.25 mmHg | 730 | 67 |

My estimates from the records imply much larger true effects: the insulin kernel accumulates to about −10 mg/dL per sustained unit/h, and norepinephrine to about +10 mmHg per mcg/kg/min.

**What this means.** The model learned that doses go with readings, but it learned almost none of the drugs' causal effect. In this ICU the effect is real, so the edit must create a dose → glucose and dose → MAP pathway of the right size and timing.

**What I will do next.**
- Open the per-stay-hour CSV to see the columns, which hours count ("reached" rows), and the per-hour shifts.
- Compute my predicted true readout on exactly these stays and hours, using the recorded doses and my kernels. That gives the concrete targets.

**What would count as evidence.** If the true insulin ×2 target is around −8 to −12 mg/dL and the model gives −0.35, the gap is very large, and so is the edit needed.

```python
fw = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(fw.shape); print(fw.columns.tolist())
print(fw.head(30).to_string())
print(fw.groupby('edit' if 'edit' in fw.columns else fw.columns[0]).size())
```