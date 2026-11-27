vocab.json gives the conventions: z = (x - median)/IQR per code, clipped to ±10, and fine bin = 1 + the number of edges below x. I'll print the full numeric note, the code ids and each numeric entry's (code, attribute, median, IQR) without the edge lists. That is everything needed to build atom embeddings exactly.

```python
print(VOCAB['numeric_note'])
print([(c['id'], c['name']) for c in VOCAB['codes']])
print([(n['code'], n['attribute'], n['median'], n['iqr'], len(n['fine_bin_edges'])) for n in VOCAB['numeric']])
print([k for k in VOCAB['codes'][4].keys()])
```