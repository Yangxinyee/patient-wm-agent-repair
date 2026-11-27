`kidx` gives, for each reading, the index of the previous bundle, whose state the heads use. Readings at hour −1 map to the admission bundle at −2, and readings at hour 1 map to the drug bundle at 0.08.

My call to `view_from` failed because I don't remember its argument layout, and the kernel can't show the source. I can still inspect its bytecode metadata: argument and local names, plus the globals and constants it uses. That should show what `prev` and `dt` are meant to be (a raw state row with dt in seconds or hours, or a structured row).

```python
for fn__ in [view_from, ctx_from, ctx_of]:
    co=fn__.__code__
    print(fn__.__name__, 'args', co.co_varnames[:co.co_argcount], '| locals', co.co_varnames[co.co_argcount:], '| globals', co.co_names, '| consts', [c for c in co.co_consts if not hasattr(c,'co_code')][:20])
```