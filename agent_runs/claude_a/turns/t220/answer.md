The grid problem is found. The atom I varied has `kind` = 2 and is not treated as an action, so its vector ignores rate. It is almost certainly an infusion END atom.

Each hourly infusion apparently yields a START atom and an END atom. The END of the previous infusion falls in the same minute-5 bundle as the next START. My bundle rates took the rate from every insulin atom, so the decoder targets mixed START and END atoms. That alone can explain the poor fits (R² 0.22 even from all of u).

The norms fit this picture: u is a constant vector (norm 11.73) wherever no action atom is present, including some "insulin" bundles.

Next I will:
- read VOCAB's atom and event kinds;
- count kinds for the drug atoms;
- recompute bundle rates from START (action) atoms only;
- recheck the factual R² of rate decoding from full u and from head 3's value.

If decoding from START atoms is good, I will redo the edited datasets with START rates. Their atom vectors do depend on rate, so the re-encodings will differ this time.

```python
print('atom_kinds', VOCAB['atom_kinds'], '| event_kinds', VOCAB['event_kinds'])
for ev_ in ['insulin','norepinephrine']:
    a_=at[at.event==ev_]; print(ev_, a_['kind'].value_counts().to_dict(), {k:round(float(a_[a_.kind==k]['rate'].mean()),3) for k in a_['kind'].unique()})
av,isa=atom_vectors(at, {}); isa=np.asarray(isa).astype(bool)
print('is_action by (event,kind):', at.assign(isa=isa).groupby(['event','kind'])['isa'].mean().round(2).to_dict())
```