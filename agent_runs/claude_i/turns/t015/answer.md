The original model says insulin ×2 lowers finger glucose by 2.0 mg/dL on average (698 stay-hours, 79 stays) and ×½ raises it by 1.16. Norepinephrine ×2 raises arterial MAP by 0.37 mmHg and ×½ changes it by −0.06.

So the model already has insulin's direction right. The question is whether the size and time course are right. My next-hour estimate is about −2 per u/h, and the effect persists for several hours. If glucose integrates insulin, the true effect of doubling over hours 13–24 could be several times larger than −2. The norepinephrine effect still needs to be estimated from the records.

I'll inspect the per-stay-hour file: its columns, how the shift evolves with hours since 12, and which hours count as "reached". That tells me exactly what the readout averages.

```python
import glob
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de.shape); print(de.columns.tolist()); print(de.head(20).to_string())
for c in de.columns:
    if de[c].dtype==object: print(c, de[c].unique()[:10])
```