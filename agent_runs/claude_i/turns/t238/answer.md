The sort failed because `atoms_by_bundle` mixes int and tuple keys. It is a defaultdict, so my int lookups silently created empty entries and returned empty atom lists. That explains why the atom offset made no difference: every e/u was the empty-set encoding. The real keys are probably tuples such as (stay_id, time). I will print a few tuple keys, one value excerpt and enc2.py, and remove the int keys I created by accident.

```python
bad_keys=[k for k in list(atoms_by_bundle.keys()) if not isinstance(k,tuple)]
for k in bad_keys: del atoms_by_bundle[k]
tk=list(atoms_by_bundle.keys()); print('n tuple keys',len(tk),' examples',tk[:3])
print('value excerpt:',str(atoms_by_bundle[tk[1]])[:300])
print(open('/work/enc2.py').read()[:3300])
```