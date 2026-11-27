`vocab.json` has the config: 4 heads, 4 slots, memory of 128, 4 z samples at evaluation. Its other keys are codes, event and atom kinds, numeric normalisation, and bin centres and notes. For the head-level edit I need the code indices, the glucose and MAP bin centres (the shift in predicted mean from tilting the logits depends on them), and the bin note. I'll print those compactly.

```python
print('codes',V['codes']); print('event_kinds',V['event_kinds']); print('atom_kinds',V['atom_kinds'])
print('numeric keys',list(V['numeric'].keys())[:12]); print('numeric_note',str(V['numeric_note'])[:300])
vbc=V['value_bin_centres']; print(type(vbc).__name__, (list(vbc.keys()) if isinstance(vbc,dict) else len(vbc)))
print('value_bin_note',str(V['value_bin_note'])[:300])
```