The original model predicts almost no drug effects. Insulin ×2 gives −0.53 mg/dL and ×1/2 gives +0.09. Norepinephrine ×2 gives −1.33 mmHg and ×1/2 gives −0.48 MAP (both negative, so not even a consistent dose response). The records imply a clearly negative insulin effect, so the model has largely failed to learn what insulin does. It needs repair.

The readout averages over "reached" stay-hours (55 stays and 484 stay-hours for insulin; 21 stays and 238 stay-hours for norepinephrine/map_arterial). Next I will inspect the per-stay-hour file to pin down exactly which hours count. I can then compute the records-based "true" readout on the same stays with my estimated impulse response, for a like-for-like comparison.

```python
d1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(d1.shape); print(d1.columns.tolist()); print(d1.head(30).to_string())
print(d1.describe(include='all').T.to_string())
```