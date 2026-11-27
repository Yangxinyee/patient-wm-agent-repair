The first 32 outputs are confirmed as the mean, yet the error stays at 0.59. My old full-step checker used a tensor `X4d` and stays `sid4` (2020000003, ...), while `F4` starts at stay 2020000112. So `F4` may not be the file I think it is.

I'll check which saved file each in-memory array came from, along with their stays and bundle counts. If `X4d` belongs to other stays where the old pipeline matched well, I should validate there. That would also suggest `stay_bundles` is wrong for `F4`'s stays (for example, train versus tuning split).

```python
f3 = np.load('/work/fwd_3/states_original.npz'); f4 = np.load('/work/fwd_4/states_original.npz')
print('X4d', tuple(X4d.shape), 'f3', f3['x'].shape, 'f4', f4['x'].shape, 'F4==f4', F4['x'].shape == f4['x'].shape and np.allclose(F4['x'], f4['x']))
print('f4 stays', np.unique(f4['stay_id'])[:6], 'f3 stays n', len(np.unique(f3['stay_id'])), 'sid4', list(sid4)[:4])
print(json.load(open('/work/fwd_4/summary.json')) if len(open('/work/fwd_4/summary.json').read()) < 600 else open('/work/fwd_4/summary.json').read()[:600])
```