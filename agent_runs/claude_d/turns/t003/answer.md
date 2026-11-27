The vocab gives per-code robust scores and 16 fine bins. The architecture fixes where a dose edit acts: the insulin atoms' z_dose and z_rate (via `embed.numeric` columns 2 and 4), their dose and rate bins (`embed.bins.dose` and `embed.bins.rate`), then `encode_action` → u_k. Note that `last_numeric` has 8 input columns, not the 4 named in the doc.

Next I will print the full code list and the numeric tables for the drugs (insulin and norepinephrine rate and dose edges). I will also load the training and tuning records and look at their structure: event counts, the timing of infusions and readings, and how doses vary.

```python
for c in V['codes']: print(c)
for r in V['numeric']: print(r)
tr = pd.read_csv('/data/train.csv'); tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape); print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.dtypes)
```