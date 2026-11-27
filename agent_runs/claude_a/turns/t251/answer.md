`z32` reads a global named `num`, the numeric-normalisation table. The error says it is now a list, but it is being indexed with a tuple key. The `[(event, field, mean, sd)]` list printed in run 220 looks like `num` was rebuilt as a list then. I'll inspect `z32`'s arguments and constants, the current `num`, and `VOCAB['numeric']`, to rebuild the dictionary it expects.

```python
c=z32.__code__; print(c.co_varnames[:c.co_argcount], c.co_consts)
print(type(num).__name__, str(num)[:200])
vn=VOCAB['numeric']; print(type(vn).__name__, str(vn)[:400])
```