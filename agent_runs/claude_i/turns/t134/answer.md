`head_inputs` expects a prepared reading table with a `pk` column, presumably the index of the previous bundle. My kernel still has the lists `pk`, `dts`, `lh`, `lv`, `ok` and a DataFrame `gdf`. Those look like that prepared table and its columns from the earlier session.

Next I will check `gdf`'s columns, call `head_inputs(gdf, X3)` and then `head_value2`, and compare the result with the tool's predicted means.
- **If the error is about 1.3:** I can diff its feature construction against mine to find the detail I am missing.

```python
print('gdf', gdf.shape, list(gdf.columns))
try:
    hi_res = head_inputs(gdf, X3)
    print([(type(a_).__name__, getattr(a_,'shape',None)) for a_ in (hi_res if isinstance(hi_res,(tuple,list)) else [hi_res])])
    hv = head_value2(gdf, *hi_res)
    print('head_value2 ->', type(hv).__name__, [(type(a_).__name__, getattr(a_,'shape',None)) for a_ in (hv if isinstance(hv,(tuple,list)) else [hv])])
except Exception as ex:
    print('ERR', type(ex).__name__, str(ex)[:300])
```