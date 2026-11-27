`o_best` (keys unit, rate_unit, side, end_attrs) looks like the atom-encoding options that gave the exact match. Next I run `test_step` with it on one stay and check that the local core still matches the saved states. I also probe what `enc_bundle` returns, so I can write my own stay runner.

```python
print(o_best)
try:
    r = test_step([sid0], o_best, kmax=4); print('test_step ->', type(r).__name__, str(r)[:400])
except Exception as ex: print('ERR test_step', repr(ex)[:300])
try:
    r2 = enc_bundle(bb0[1][1]); print('enc_bundle ->', type(r2).__name__, [ (type(x).__name__, getattr(x,'shape',None)) for x in (r2 if isinstance(r2,(tuple,list)) else [r2])])
except Exception as ex: print('ERR enc_bundle', repr(ex)[:300])
print(bb0[1], bb0[3])
```