A shared register makes no difference, so the register is per code, as documented. Readings with an empty register are accurate for MAP but not for glucose, even at the same hour and from the same state. So part of the remaining error is code-specific. The rest is in the 8-column last-register input when the register is filled.

In restart 3 I had searched over assignments of features to those 8 columns, and the stored per-code results (`best_ty`) show errors of 1.3–1.6, better than now. Next I print the stored assignment results (`asg`, `asg2`, `best_s`, `best_c`, and the feature dictionaries) to recover the best layout found then. If they give a layout, I plug it into my head and re-check the errors against the tool.

```python
for nm_ in ['asg','asg2','best_s','best_c','cur','assign','a2']:
    if nm_ in globals(): print(nm_, globals()[nm_])
print('feats', list(feats.keys()) if isinstance(feats,dict) else feats)
print('feats2', list(feats2.keys()))
print('stat_opts', {k:(np.asarray(v).shape if hasattr(v,'__len__') else v) for k,v in stat_opts.items()})
print('part', {k:(np.asarray(v).shape if hasattr(v,'__len__') else v) for k,v in part.items()} if isinstance(part,dict) else part)
```