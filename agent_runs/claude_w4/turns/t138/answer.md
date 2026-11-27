The helpers exist: `meas_parts(code, val)` likely builds a measurement atom, and `enc_e(atoms)` applies the bundle encoder. To build e_k at reading bundles I also need the raw tuning events for the sel100 stays. So I'll list the DataFrames in the kernel and test what `meas_parts` returns for one glucose reading.

```python
dfs = {k: v.shape for k, v in globals().items() if isinstance(v, pd.DataFrame)}
print(dfs)
try:
    a = meas_parts('glucose_finger', 150.0); print(type(a), getattr(a,'shape',None) if not isinstance(a,(tuple,list)) else [getattr(x,'shape',x) for x in a][:6])
except Exception as ex: print('meas_parts err', ex)
```