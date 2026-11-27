These configs show the earlier session was matching the encoder and z posterior. The best unit-index choice (cfg_: MAP unit 5, glucose 4) gives a z-posterior error of about 0.05, close but not exact. The time-unit searches never got an exact match either. An exact full re-implementation could take many more runs.

A cheaper route may work. If the model's state already encodes the recent insulin and norepinephrine dose history in a linearly readable form, I can add a head-side correction for the missing per-step effect. Drawn readings fed back in a rollout would then carry it forward. To test that, I'll open the fwd_3 states file, see its keys and bundle labels, and confirm X3, S3 and H3 match it. A linear-readability check would come right after.

```python
z3 = np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k: (z3[k].shape, z3[k].dtype) for k in z3.files})
for k in z3.files:
    if z3[k].ndim == 1 and z3[k].dtype.kind in 'OUSb':
        print(k, z3[k][:6])
print(json.load(open('/work/fwd_3/summary.json')) if os.path.getsize('/work/fwd_3/summary.json') < 1500 else 'big')
```