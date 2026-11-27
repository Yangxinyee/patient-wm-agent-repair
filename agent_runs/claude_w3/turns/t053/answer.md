The states at insulin bundles encode the band well (R² 0.90 for the base dose) but barely encode the actual dose. R² for the rate is 0.65 and for the jitter deviation only 0.09–0.20.

So the original model hardly carries the dose magnitude in its state. That is why doubling the dose changes almost nothing downstream. A head-only edit cannot read a dose signal that is absent. The edit has to create the dose signal on the action side and carry it, persisting about 12 hours, to the glucose head. In the core, h (which knows the band) and u_k (the dose) meet only in the prelude and blocks.

Designing that injection blind, with only 47 forward passes, is very risky. A local copy of the model checked against the fwd_3 states would let me test edits cheaply. The main obstacle is tokenisation: how atoms map onto the categorical tables, kinds, bins and z-scores.

Next I'll print the vocab's code list, event and atom kinds, and normalisation notes, compactly. If the vocab specifies these mappings, the reimplementation is feasible. If not, I'll need a search over guesses, checked against the z-posterior means in the states.

```python
print(VOCAB['codes']); print(VOCAB['event_kinds']); print(VOCAB['atom_kinds'])
print(VOCAB['numeric_note']); print(VOCAB['value_bin_note'])
nm0 = VOCAB['numeric']; print(type(nm0), list(nm0.keys())[:20] if isinstance(nm0,dict) else len(nm0))
k0 = list(nm0.keys())[0] if isinstance(nm0,dict) else None
print(k0, json.dumps(nm0[k0])[:600] if k0 else '')
```