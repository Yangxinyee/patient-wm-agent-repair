For scale: the dumped z-posterior means have sd about 0.65, mean |z| about 0.17, and some dimensions have sd as low as 0.06. A residual error of 0.08 is therefore a real mismatch, not rounding noise.

My bundle table also skips some row indices (1, 5, 8, 10), so I may have dropped atoms such as infusion-END records that the model actually sees. I'll compare the raw training-file rows for this stay with the bundle table.

```python
print(rd[:1200])
tr_raw = pd.read_csv('/data/train.csv', nrows=400)
print(tr_raw.columns.tolist())
r_s = tr_raw[tr_raw.iloc[:,0]==tr_raw.iloc[0,0]]
print(r_s.head(12).to_string()[:1400])
```