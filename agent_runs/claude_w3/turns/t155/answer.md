Using the same row is no better (0.75 with the flow, 0.60 without), so that convention is wrong. My old `zpost_err('own')` reached 0.10, so it must do something my fresh code doesn't. I'll list the global names it references (data arrays and helpers) to identify the difference.

```python
c = zpost_err.__code__
print([n for n in c.co_names][:60])
print(c.co_varnames[:40])
```