The helpers are confirmed. `tf_stay` (teacher-forced) returns per-reading predictions and, at each drug bundle, the context vector, u, and the insulin and norepinephrine rates. `tf_stay_z` and `bstep_z` take a custom z-token function. `rollout` is the offline replica of the tool's dose_edit.

The cleanest approach is to write the correction as real weights in z_post and z_in. The standard `bstep` then runs it, and so do `tf_stay` and `rollout`, with no special hooks.

Before I collect training data I need three things:
- how bundles are built for any stay (`setup.py`),
- the rollout's interface,
- what the main global containers hold (weights dict, bundles, data frames).

I'll print those compactly.

```python
print(open('/work/setup.py').read())
print('-----rollout')
print(open('/work/wm2_rollout.py').read()[:4300])
for k in ['W0','BUN','Dtr','Dtu','stays','pol','Ptu','X0row','u_empty','MEASC','CENd','DRUGN','tb','th','sidx_tu']:
    v=globals().get(k,None)
    try: ln=len(v)
    except Exception: ln=None
    print(k, type(v).__name__, ln, (list(v.columns)[:12] if hasattr(v,'columns') else ''))
print([n for n in NAMES if n.startswith('z_')])
```