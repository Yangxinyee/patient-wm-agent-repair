The vocab output was cut off near the end of the tensor list. Insulin rates appear to come in steps of 0.5 U/h, with median 1.5 and IQR 1.0, so z = rate − 1.5. Doubling a 3 U/h rate gives z = 4.5, which may be at the edge of the training distribution. Norepinephrine rate edges top out at 1.0 mcg/kg/min, so that may be a cap. Next I will print the rest of the tensor names and load the training records to see their structure: event counts, the timing of drugs and readings, and how many stays have insulin or norepinephrine.

```python
idx = NAMES.index('gap.0.bias')
for n in NAMES[idx:]: print(n, SHAPES[n])
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.dtypes)
```