fwd_4 holds full per-bundle states (644 bundles from 12 stays), and `BUN` holds my bundle tables for those stays. The old check took the dumped state of the previous bundle, rebuilt e_k and u_k, and compared the z-posterior mean. Its error was about 0.08, and several encoder variants gave similar errors. That points to a mismatch in the encoder or the posterior, not in the context.

To see the scale of that error, I'll inspect one bundle table and the spread of the dumped z-posterior means.

```python
s_ = stays12[0]
b0 = BUN[s_]
print(type(b0), getattr(b0, 'shape', None))
print(b0.head(8).to_string()[:1500])
zz = F4['x'][:, 960:992]
print('z post mean: overall sd %.3f, mean|z| %.3f' % (zz.std(), np.abs(zz).mean()))
print('per-dim sd (first 8):', np.round(zz.std(0)[:8], 3))
```