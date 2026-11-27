The vocab keeps its normalisation and bin edges under `numeric`, with a `numeric_note` explaining the convention. Neither was printed because they aren't dicts keyed by code name. I'll print the note, the codes, and the insulin and norepinephrine entries of `numeric`.

Then I'll check whether an insulin rate edge sits between 1.5 and 2.0, and whether the bins are assigned as x < edge or x ≤ edge. If the note's convention differs from the side my rebuild uses, that explains the misplaced jump.

```python
print('numeric_note:', VOCAB['numeric_note'])
print('codes:', json.dumps(VOCAB['codes'])[:800])
num_=VOCAB['numeric']; print(type(num_), (list(num_.keys()) if isinstance(num_,dict) else len(num_)))
items_=num_.items() if isinstance(num_,dict) else enumerate(num_)
for k_,v_ in items_:
    s_=json.dumps(v_)
    if ('insulin' in str(k_)) or ('norepi' in str(k_)) or ('insulin' in s_[:200]) or ('norepi' in s_[:200]): print(k_, s_[:1500])
```