Two things simplify the reproduction:
- Value bins are per code, with centres in raw units, and the predicted mean is Σ p(b)·centre[b−1].
- The 15 `type_emb` rows most likely encode event kind (5) × atom kind (3).

The unknowns to settle against the factual CSV are the type index for readings, the 8 features of `last_numeric`, the scaling inside `gap`, and how the z prior enters. Next I'll print the code ids and the data README, and look at one stay's raw rows. I need these to rebuild bundles that line up exactly with the state rows.

```python
print([(c['id'],c['name']) for c in V['codes']])
R_ = open('/data/README.md').read(); print(len(R_)); print(R_[:2500])
TRd = pd.read_csv('/data/train.csv', nrows=40); print(TRd.columns.tolist()); print(TRd.head(12).to_string()[:1500])
```