I have the signatures but not the code. Next I'll look at the compiled code objects of gated3, pair, fit_ro5 and eval_edit: the global names they use, their local variables and their constants. That should show how the edit is put together (which readouts, gates, units and logit directions), so I can swap the NE part for a level readout.

```python
for fn in ['gated3','pair','fit_ro5','eval_edit','lda2','lda_in']:
    c = globals()[fn].__code__
    print(fn, 'names:', c.co_names[:60])
    print('   vars:', c.co_varnames[:40])
    cs = [x for x in c.co_consts if not hasattr(x,'co_code')]
    print('   consts:', str(cs)[:400])
```