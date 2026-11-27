`run_stay` takes per-bundle minutes, dt and the memory of e vectors (including the current one) with ages in seconds. So I can write a runner that adds a chosen Δz at selected drug bundles.

I also need a head that gives each reading's predicted mean from the state. Rather than rewrite it, I'll check whether the existing `pred_mean_s` works on the factual table F. F should hold the tool's predicted means for these same training stays. If `pred_mean_s` reproduces the tool's means with a typical gap of about 1 mg/dL (the prior-z sampling noise noted in session 2), I can use it, with common random numbers, to measure how a z injection changes glucose and MAP predictions.

```python
print(F.columns.tolist()); print(F.head(3).to_string()[:800])
try:
    pm_ = pred_mean_s(np.arange(40))
    print(type(pm_), np.shape(pm_))
    print(np.round(np.asarray(pm_)[:10],2))
except Exception as ex:
    import traceback; traceback.print_exc(limit=2)
```