That output was wasted: `prelude`'s default argument is a whole weight dictionary, and printing it filled the 10,000-character limit before anything useful appeared. I did learn that `prelude` takes the token stack TOK, the prelude weight dictionary W (default = the original weights) and a flag `ret`. I also saw that the per-dimension decay `flow.a` runs from about −6.4 to −18, so time constants span from minutes to years.

Next I'll repeat the inspection without the defaults:
- the global names `prelude` uses and its short-form bytecode;
- the arrays in memory sized to the 2,319 drug bundles (or 520 or 300), and whether the step replica's targets and inputs (yt, ot, htf-like arrays) still exist.

This decides whether I can compute the injection Jacobian on the CPU or must measure it with probe edits.

```python
c=prelude.__code__; print('prelude args', c.co_varnames[:c.co_argcount], 'n_defaults', len(prelude.__defaults__ or ()), 'default types', [type(d).__name__ for d in (prelude.__defaults__ or ())])
print('names', c.co_names); print('locals', c.co_varnames)
t=brief(prelude)
print(' ; '.join(x.split(' ',1)[-1] if x.startswith(('LOAD_FAST','LOAD_GLOBAL','LOAD_ATTR','LOAD_CONST','STORE_FAST')) else x.split(' ')[0] for x in t)[:2500])
arrs=[]
for k,v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v,(np.ndarray,torch.Tensor)) and v.ndim>=1 and v.shape[0] in (2319,2320,520,300):
        arrs.append(f'{k}:{tuple(v.shape)}')
print('\nARRAYS:', ' | '.join(arrs))
for k in ['yt','ot','htf','HT','Hd','Ed','TOKd','TK']:
    if k in globals(): print('HAVE',k, getattr(globals()[k],'shape',type(globals()[k])))
```