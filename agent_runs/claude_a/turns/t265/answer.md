Both stored builds cost +0.0094, so neither can be used as is. The rollout failure looks like my own fault. In runs 247–250 I assigned throwaway globals (S, L, d, g, f, keep, t, R, r, s, ed, …). Kernel functions written in earlier runs may read globals with those names, and I may have overwritten them. This run catches the rollout error's traceback (frame names and line numbers only). It also lists which kernel functions reference any name I reassigned. If a clobbered name shows up in the failing chain, I'll restore it next. If none does, the failure is in how I called the function, and the traceback should point to it.

```python
import traceback, sys, types
try:
    R=rollout_stay(tu, s0, PO, 4, 0)
except Exception as ex:
    tb=traceback.extract_tb(sys.exc_info()[2]); print('chain',[(fr.name, fr.lineno) for fr in tb][-6:], type(s0), s0)
mine=set('S L d g f keep names src t R r b0 n0 s0 fn dk ar sc nm ex dfs ed s x a b out t0'.split())
def gnames(co):
    out=set(co.co_names)
    for c in co.co_consts:
        if isinstance(c, types.CodeType): out|=gnames(c)
    return out
hits={}
for nm_ in fn:
    ob=globals().get(nm_)
    if hasattr(ob,'__code__'):
        z=gnames(ob.__code__)&mine
        if z: hits[nm_]=sorted(z)
print('clobber refs', hits)
```