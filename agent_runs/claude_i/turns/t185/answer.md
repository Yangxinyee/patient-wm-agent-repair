F is the tool's factual output for the same 100 training stays as the states. Each reading has the tool's predicted mean, its NLL, the index of the previous bundle, dt and the last value of that code. That is what I need to validate a local head.

`pred_mean_s` expects a 6-part tuple whose layout I can't read. Its code object still lists the variable names, constants and globals it uses, which should reveal the layout of the 8-column `last_numeric` input and the value-head type index. I'll inspect those for `pred_mean_s`, `view_ctx` and `zprior`.

```python
for f_n in ['pred_mean_s','view_ctx','zprior','head_inputs']:
    co_ = globals()[f_n].__code__
    print(f_n, 'args', co_.co_varnames[:co_.co_argcount], '| vars', co_.co_varnames[co_.co_argcount:][:40])
    print('   names', co_.co_names[:50])
    print('   consts', [c_ for c_ in co_.co_consts if not hasattr(c_,'co_code')][:40])
```