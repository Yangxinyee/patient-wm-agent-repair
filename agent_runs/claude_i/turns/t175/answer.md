Every target came out with standard deviation 0, so the lookups returned nothing. The cause is a key mismatch (stay-id type or event naming) or an empty tuning subset, not a property of the model. I'll check the size of `tu`, its event names and the key types before re-running the decodability test.

```python
print(len(tu), tu.event.unique()[:10], tu.stay_id.dtype, type(S3[0]))
print(list(Rins.items())[:3], len(Rins), len(Rne))
print(tune.stay_id.dtype, tune.stay_id.iloc[:2].tolist(), S3[:2], np.isin(S3[:5], tune.stay_id.unique()))
```