The undocumented work left four files dated after the last notes save:
- tf.py and tf2k_orig.parquet (11:59): probably a teacher-forced pass of the original model on 2,000 training stays.
- cf.py (12:25), rollout_h.py (12:34), oracle.py and policy.py (12:47): probably counterfactual and "oracle" experiments.

I need their contents to know whether runs 150 to 182 already tested an edit mechanism. This run prints four of the files; I'll read rollout_h.py next. Nothing is executed or changed.

```python
for f in ['tf.py','cf.py','oracle.py','policy.py']:
    s=open('/work/'+f).read()
    print('=====',f,len(s)); print(s[:3000])
```