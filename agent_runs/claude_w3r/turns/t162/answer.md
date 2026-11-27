With an intercept term, the norepinephrine correction improves the estimated factual fit by about 0.005 nats per reading. With β1 = 3.45, β2 = 0.72 fixed it is still −0.005. Last MAP gets almost no weight, but the fit wants a negative r3 term, which absorbs the level. So a lag-1/lag-2 norepinephrine channel plus a constant is factually affordable.

Now I need the interventional targets for both drugs on a common set of stays, computed exactly like the readout. Next I'll read fwd_1's dose_edit CSV and summary to learn the aggregation, i.e. which stay-hours count and whether averaging is per stay first. Then I'll compute records-based targets from my structural IRFs on those rows, for all four edits.

```python
D1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(D1.columns.tolist(), D1.shape); print(D1.head(4))
S1 = json.load(open('/work/fwd_1/summary.json')) if 'json' in globals() else __import__('json').load(open('/work/fwd_1/summary.json'))
print(str(S1)[:1200])
```