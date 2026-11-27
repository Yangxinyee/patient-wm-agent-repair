The README shows that each infusion record has a rate, a dose and units, with a matching END at end_hour. The vocab holds the codes, event kinds, atom kinds and per-code numeric normalisation. Next I will print these vocab entries compactly. They define the robust z-scores and fine bins that turn a dose into its embedding. From them I can rebuild an insulin atom's embedding. That would confirm the action encoder can be reimplemented, apart from a few constant categorical terms.

```python
print(VOCAB['codes']); print(VOCAB['event_kinds']); print(VOCAB['atom_kinds'])
print(VOCAB['numeric_note']); print(VOCAB['value_bin_note'])
for e_ in VOCAB['numeric']:
    print({k_: (np.round(v_,3).tolist() if isinstance(v_,(list,float)) else v_) for k_,v_ in e_.items()})
```