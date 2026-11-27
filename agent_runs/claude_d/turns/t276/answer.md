Run 248 failed inside `atom_vec` because I overwrote a global it relies on. I most likely redefined `Wt` (which was probably a weight dict) as a function. I will list the globals `atom_vec` and `uenc` reference and restore the clobbered name as a dict of weights. Then I rerun the same u(r) analysis with non-clashing helper names. The question is unchanged: does the encoding of a doubled NE rate above 1.0 leave the in-range dose trend, and does capping the bin restore it?

```python
print('atom_vec names', atom_vec.__code__.co_names); print('uenc names', uenc.__code__.co_names)
clob=[n for n in set(atom_vec.__code__.co_names+uenc.__code__.co_names) if n in ('Wt','gel','mlpP','lnP','uenc2','aS','aE','E','G','R','dec','B','h','m','g','A','N','D11','TU','gam','X','b','d')]
print('possibly clobbered', clob)
```