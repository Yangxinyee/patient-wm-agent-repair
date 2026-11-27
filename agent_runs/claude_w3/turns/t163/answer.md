The atom-level encoder gives the same error, so the encoders aren't the problem. The old `zpost_err` must build the context differently: zeroing e there only moves its error from 0.10 to 0.19, so its context alone fits well. I'll disassemble `zpost_err` and print a compact trace of the names and operations it uses after the encoding. The goal is to recover its context formula.

```python
import dis
ins_ = [(i.opname, i.argrepr) for i in dis.get_instructions(zpost_err)]
toks = [a if a else o.split('_')[0] for o, a in ins_ if o in ('LOAD_GLOBAL','LOAD_ATTR','LOAD_FAST','STORE_FAST','BINARY_OP','LOAD_CONST','BINARY_SUBSCR','CALL') and (a or o in ('BINARY_SUBSCR',))]
s_ = ' '.join(t for t in toks)
i0 = s_.find('out_p')
print(len(s_)); print(s_[i0-200:i0+1600])
```