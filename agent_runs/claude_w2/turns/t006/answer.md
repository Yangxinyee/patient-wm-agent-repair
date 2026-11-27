Insulin rates come in 0.5-unit steps, mostly between 0.5 and 3 U/h. Norepinephrine rates come in 0.01 steps, and the dose bin edges saturate at 1.0, which suggests a cap. For insulin dose equals rate, but for norepinephrine dose/rate varies (the maximum gap is 3.8). Since NE dose = rate × 60 min × weight / 1000, the dose carries patient weight, which the model can see. Insulin is used in about 61% of stays and norepinephrine in about 43%; many stays have it for the whole 25 hours.

Next I'll check four things:
- the stay_id structure, since a prefix might encode a unit or protocol arm and give me a natural instrument;
- the range of NE rates (is there a 1.0 cap?);
- the implied weights (NE dose/rate/0.06);
- an hourly per-stay grid of readings and rates, which is the basis for every causal analysis.

```python
s = pd.Series(tr.stay_id.unique())
print(s.head(10).tolist(), s.tail(10).tolist())
print((s // 10**7).value_counts().sort_index().head(20))
print((s % 10**7).describe())
st = pd.Series(tu.stay_id.unique()); print(st.head(5).tolist(), (st//10**7).value_counts().sort_index())
ne = tr[tr.event=='norepinephrine']
print('NE rate max', ne.rate.max(), (ne.rate>=1.0).mean())
w = (ne.dose/ne.rate/0.06)
print(w.describe())
print(ne.assign(w=w).groupby('stay_id').w.std().describe())
```