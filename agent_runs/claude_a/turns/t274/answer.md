The rate lookup returned all zeros, so `'ins'` isn't the drug key `bundle_rates` expects. The ridge regression had no target variance, and the builder then divided by zero. A second wrong assumption: the dose-encoder output is nonzero at every bundle (13,036 of 13,036), so bundles without a dose get a constant "empty" encoding rather than zeros. I'll print the event names in the records, the string constants `bundle_rates` uses, and the columns of the bundle table, so I can call it with the right key and pick out dose bundles correctly. The aim is a per-bundle insulin-rate vector in which roughly 10–20% of bundles are nonzero.

```python
print('events', list(tr['event'].unique())[:12])
print('bundle_rates consts', [c for c in bundle_rates.__code__.co_consts if isinstance(c,str)][:12], bundle_rates.__code__.co_varnames[:8])
bd=dsE['bd']; print('bd cols', list(bd.columns)[:15], len(bd))
at=dsE['at']; print('at type', type(at).__name__, (list(at.columns)[:12] if hasattr(at,'columns') else (len(at) if hasattr(at,'__len__') else '')))
```