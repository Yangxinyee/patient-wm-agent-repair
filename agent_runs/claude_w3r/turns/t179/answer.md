`var` is a dict of options indexed by string keys. I'll read the string constants in `core_step`'s code object to find those keys, and the defaults used in run_stay calls. Then I can call it with the validated configuration.

```python
c_ = core_step.__code__
print('consts:', [x for x in c_.co_consts if isinstance(x, str)][:40])
print('names:', c_.co_names)
print('OPTS', OPTS)
print('PA keys', list(PA.keys()) if isinstance(PA, dict) else type(PA))
```