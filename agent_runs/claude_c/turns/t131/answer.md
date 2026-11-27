The states have 4356 bundles across the 100 stays. Before linking them to the exogenous within-tier dose noise, I need the record layout of `tuning.csv`.

```python
tu=pd.read_csv('/data/tuning.csv')
print(tu.columns.tolist()); print(tu.shape)
print(tu.head(6).to_string()[:1500])
```