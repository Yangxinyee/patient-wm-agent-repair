The vocab has insulin as code 8 and NE as code 9, with numeric normalisation and bins in `VOCAB['numeric']`. Before building the gated mlp1 correction I need to know which NE atoms carry the doubled rate in a rollout bundle. A drug record makes a START atom at its hour and an END atom at end_hour, and my notes only verified the START-atom formula. The "u-path / e-path" search in runs 237–238 (best error about 0.1) was probably about how END atoms are encoded, and was unresolved. If the kernel keeps an input history (`In` / `_ih`), I can read back exactly what runs 236–238 did instead of re-deriving it. I am checking that, along with the signatures of the remaining helpers.

```python
h=globals().get('In') or globals().get('_ih')
print('history', type(h), (len(h) if h is not None else None))
for f in ['ctx_of','bundles_of','items_of','step_of','zerr','zerr_e','avec2','catsE','f_ne','zpr','zs_','zsc']:
    c=globals()[f].__code__; print(f, c.co_varnames[:c.co_argcount], globals()[f].__defaults__ if not isinstance(globals()[f].__defaults__,tuple) or len(str(globals()[f].__defaults__))<80 else '...')
nm=VOCAB['numeric']; print(type(nm), (list(nm.keys())[:10] if isinstance(nm,dict) else nm[:2]))
print(VOCAB['numeric_note'][:400])
```