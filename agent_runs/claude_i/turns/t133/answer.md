My rebuilt head is close but not exact. The best setting (type row 1, the 8 inputs as [has_last, age, z_last, has_value, 0, …]) gives a mean absolute difference of 4.0 mg/dL from the tool. My earlier session's version reached about 1.3, which is roughly the tool's own z-sampling noise. So one detail is still wrong; it could be the z handling, the age or value features, or how the mean is formed.

That earlier version is still in the kernel as `head_inputs(F, X)` and `head_value2(...)`. Its source is gone, but I can call it. Next I will run `head_inputs` on the fwd_2 readings with the fwd_3 states, inspect what it returns, and check its predictions against the tool.
- **If it gives MAD ≈ 1.3:** I can compare its intermediate inputs with mine to find the difference.

```python
try:
    hi_res = head_inputs(F2s, X3)
    print(type(hi_res).__name__, [ (type(a_).__name__, getattr(a_,'shape',None)) for a_ in (hi_res if isinstance(hi_res,(tuple,list)) else [hi_res])])
except Exception as ex:
    import traceback; print('head_inputs ERR', type(ex).__name__, str(ex)[:300])
```