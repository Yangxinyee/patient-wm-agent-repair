The head tensors match the architecture description. Value head:
- a 192→192→17 MLP on `with_code`,
- plus a transition table indexed by code·17 + last value bin.

Two details differ from the description: `last_numeric` takes 8 inputs, not 4, and `type_emb` has 15 rows. So before writing the replica I need the vocabulary notes on how the register numbers are built, which type index a value query uses, and the bin edges and centres.

Next I'll print those notes, the glucose_finger numeric entry, the flow‑decay tensor's shape and the embedding table shapes. This run is just gathering the information needed for an exact replica, so there is no hypothesis to test yet. The test comes in the following run: does the replica reproduce fwd_2's predicted means?

```python
for k in ['event_kinds','atom_kinds','numeric_note','value_bin_note']:
    print(k,':',json.dumps(VOCAB[k])[:900])
nu=VOCAB['numeric']; print(type(nu), list(nu.keys())[:20] if isinstance(nu,dict) else len(nu))
gf=nu['glucose_finger'] if isinstance(nu,dict) and 'glucose_finger' in nu else (nu[0] if isinstance(nu,list) else None)
print(json.dumps(gf)[:1500])
print('centres keys',list(VOCAB['value_bin_centres'].keys())[:20] if isinstance(VOCAB['value_bin_centres'],dict) else len(VOCAB['value_bin_centres']))
for n in NAMES:
    if n.startswith('embed') or 'flow' in n or n.startswith('core.gap') or 'slots' in n: print(n,SHAPES[n])
```