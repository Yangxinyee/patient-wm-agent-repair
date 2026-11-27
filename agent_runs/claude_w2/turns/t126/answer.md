The core step is now exact as well. h differs by 0.00006 on a scale of 25, and `out` is exact to five decimals. The memory includes the current bundle's e, and the age embedding core.age(log(1+age in seconds)) is added to the memory vectors before the key/value projection.

The last piece is the value head. In restart 3 I matched it against the factual predicted means (functions `f8_`, `run13`) but did not record the final configuration, in particular the layout of the 8 last-register numeric columns and the view's gap term. Next I decompile `f8_` and `run13`, so I can rewrite the head into impl.py. If they show the column layout and the prior-z averaging, I can then verify the head against the fwd_4 factual means and NLL.

```python
compact(f8_)
print('-----')
compact(run13)
```