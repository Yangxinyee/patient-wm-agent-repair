fwd_3 holds the original model's states after each of 5,156 bundles in 100 training stays. Each row has the bundle hour, whether it held insulin or norepinephrine, and the 992-number state (out, 4 slots of h, z posterior mean).

The question is whether recent doses can be read linearly from the state. For each bundle I'll build features from the record: the insulin and norepinephrine rates started in each of the past 1..6 hours. Then I'll cross-validate a ridge regression of each feature on the state, grouped by stay. First I'll check the record's column format so the features come out right.

```python
print(train.columns.tolist())
print(train[train.event=='insulin'].head(3).to_string())
print(train.event.value_counts().to_dict())
st3 = np.unique(z3['stay_id']); print(len(st3), st3[:5], np.isin(st3, train.stay_id.unique()).all())
```