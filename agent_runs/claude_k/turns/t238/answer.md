Run 221 failed inside my old encoder helper. The function chain `u_rec → setenc_my → mlpf` calls a global name that I overwrote with a list this session. The likely culprit is `L`, which I reused in run 203 for the doc's lines.

I can't print the helper's source, but I can read each function's referenced global names from its code object. Next I'll list those names, flag any that now hold a list, and look at the defaults and constants that `mlpf` uses. Then I can redefine the overwritten global correctly instead of rewriting the encoder from scratch.

```python
for fn_n in ['mlpf','setenc_my','u_rec','datom','mk_best','mk']:
    f_=globals().get(fn_n)
    if f_ is None: print(fn_n,'missing'); continue
    cn=f_.__code__.co_names; bad=[c for c in cn if isinstance(globals().get(c),list)]
    print(fn_n,'globals:',cn[:25],'| now lists:',bad)
print('mlpf consts',mlpf.__code__.co_consts[:12],'args',mlpf.__code__.co_varnames[:mlpf.__code__.co_argcount])
```