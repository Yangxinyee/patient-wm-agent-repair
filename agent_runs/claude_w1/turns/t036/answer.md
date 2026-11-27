The residual test is too noisy to be useful, with only 1,391 finger readings. The lag-1 shock coefficient is −3.4 (SE 0.74) and lag-2 is −1.7 (SE 0.8). That is consistent with the model missing most of the causal slope. But the expected-dose term at lag 1 is also −2.2 (SE 0.7), and later lags are all within noise. Getting SEs near 0.2 this way would take about 12 more factual passes, which is too expensive.

The stronger route is to reproduce the model locally on CPU. Then I can compute factual predictions and internal states for thousands of training stays, and test candidate edits offline, including rollouts. The architecture doc leaves gaps: atom fields, categorical tables, `embed.kq`, activations, norms. Next I'll read `tensors.json` for per-tensor descriptions, and look at the scale of the small embedding tables, to infer how atoms are built.

```python
tj = json.load(open('/model/tensors.json'))
print(type(tj), len(tj))
s = json.dumps(tj)[:3000]
print(s)
```