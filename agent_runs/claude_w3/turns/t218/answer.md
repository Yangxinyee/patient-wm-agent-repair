My fresh rebuild of the z-posterior is far off: median relative error 0.62 on bundles without drugs and 0.35 on drug bundles. That is much worse than the 0.10 my older helper `zpost_err` reported, so some detail in the new code is wrong. The likely suspects are the time-gap units or how 'hour' is stored. Next I will check the fractional parts of the hour column and the names and constants that the older helper uses, so I can copy its conventions. If 'hour' turns out to be rounded, or the older helper uses another dt unit, that explains the gap.

```python
fr=np.round((hr4%1)*60,2); print('minute-of-hour values', np.unique(fr)[:12])
c=zpost_err.__code__; print(c.co_names); print([k for k in c.co_consts if not hasattr(k,'co_code')][:30])
print('zpost_err():', zpost_err())
```