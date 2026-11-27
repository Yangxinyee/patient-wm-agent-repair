`memfill` selects from a dictionary of memory-filling rules. `'e'` comes closest (median error 0.82) but isn't exact, so the memory holds something other than the plain measurement vector. The rules are defined in nested code objects inside `run_variant3`. I'll print their names and constants to see the available options and what each stores.

```python
for c_ in run_variant3.__code__.co_consts:
    if hasattr(c_,'co_code'):
        print(c_.co_name, 'names', c_.co_names[:12], 'consts', [str(x)[:14] for x in c_.co_consts][:12], 'vars', c_.co_varnames[:6])
```