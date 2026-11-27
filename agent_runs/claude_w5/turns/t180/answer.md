`cstep` builds h_new (local `hn`) and calls four helpers: `ln_`, `mlp_`, `lin_` and `att_`. Its constants list the weight keys it uses, in order.

Next I will read the helpers' signatures and argument names. Then I can write `cstep_h`, a copy of the core step that also returns h_new and the input to the coda, and check that it reproduces `cstep` exactly.

It will be confirmed if the helpers take (input, weight-prefix) style arguments that I can call directly.

```python
for k in ['ln_','mlp_','lin_','att_']:
    fo=globals()[k]; c=fo.__code__
    print(k, inspect.signature(fo), '| vars', c.co_varnames, '| names', c.co_names, '| consts', [x for x in c.co_consts if isinstance(x,(str,int,float))])
```