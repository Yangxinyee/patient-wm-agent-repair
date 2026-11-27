The bias is broad and systematic, not a few runaway paths:
- At hour 13, the first hidden hour, the median stay is +15 mg/dL and 71% of stays are above +10.
- It is almost entirely in stays with insulin running at hour 12: +17 against +4 without.
- It correlates with that stay's glucose level (r = 0.67), and it grows hour by hour.

This fits the decoded dose surprise losing its dose component and keeping its negative glucose-implied-dose component. The edit then adds an upward shift proportional to the glucose level.

At hour 13 the rollout history is still the real record, so the teacher-forced prediction for those readings should show the same bias if the GPU head matches my offline head. My offline check was run only on the train stays used to fit the decoders, so it could not show a generalisation failure.

Next step: a factual request with both models on the same 100 tuning stays. It is the binding factual-fit check anyway.
- If the teacher-forced hour-13 bias is about +17 too, the head itself misbehaves on new stays. That would be overfitting or a mismatch between GPU and offline, and the decoder design must be redone.
- If it is near 0, the drift is specific to rollouts.

First I need the stay IDs written out for the request.

```python
sl_ = json.load(open('/work/sel1.json'))
k_ = list(sl_.keys())[0]; ids_ = sl_[k_]
print(k_, len(ids_)); print(json.dumps([int(x) for x in ids_]))
```