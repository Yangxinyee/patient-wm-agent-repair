The memory options are `'e'`, `'zero'`, `'u'` and `'e+u'`. `'e'` and `'zero'` don't match. I'll test `'u'` and `'e+u'` over the full stay. The one with errors around 1e-4 or less is the correct memory rule. If neither matches, the mismatch has another cause, such as skip handling or ages.

```python
for mf in ['u','e+u']:
    r_=np.asarray(run_variant3(mf, kmax=len(BUN[st_]), stay=st_))
    print(repr(mf), 'n', r_.size, 'max', float(r_.max()), 'median', float(np.median(r_)))
```