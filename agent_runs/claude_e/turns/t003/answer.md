The vocab shows how doses are encoded. Insulin rates sit on a coarse grid (0.5, 1.0, 1.5, … up to 5 u/h). The bin edges repeat, so doubling a rate moves it several bins. Norepinephrine rates top out at 1.0 mcg/kg/min, and the upper edges all equal 1.0, so anything above 1.0 lands in bin 16. Rates are also clipped through z = (x − median)/iqr, capped at ±10.

Next I'll print the bin centres and the truncated numeric entries, then load train.csv. I want its structure: events per stay, how often each event occurs, and how doses are spread.

```python
print(json.dumps(V['value_bin_centres'])[:3000])
for e in V['numeric']: print(e)
tr = pd.read_csv('/data/train.csv'); tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape); print(tr.head(30).to_string())
print(tr.event.value_counts())
```