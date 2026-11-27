`gdf` is a small record excerpt, not the prepared table. The prepared reading table was probably `F`, which my earlier session built from the fwd_2 factual output. It should have a `pk` column (the previous bundle) along with dt, last value and age.

Next I will check whether `F` has those columns and, if it does, run `head_inputs` and `head_value2` on it. I then compare the result with the tool's predicted means, and also with my rebuilt head on the same readings.

```python
print('F', F.shape, list(F.columns)[:20])
if 'pk' in F.columns:
    hi_res = head_inputs(F, X3)
    print([(type(a_).__name__, getattr(a_,'shape',None)) for a_ in hi_res])
    hv = head_value2(F, *hi_res)
    print('hv', [(type(a_).__name__, getattr(a_,'shape',None)) for a_ in (hv if isinstance(hv,(tuple,list)) else [hv])])
```