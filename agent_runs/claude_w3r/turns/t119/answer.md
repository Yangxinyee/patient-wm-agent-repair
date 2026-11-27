My encoding uses bin = 1 + number of edges below x (searchsorted side='left') and z = (x − median)/iqr clipped at ±10. The vocab file has explicit notes on numeric conventions (numeric_note, value_bin_note) and the event and atom kind codes. Next I will print those notes and compare them with my implementation.

- **Confirms a fixable mistake:** a note contradicts my choices, for example on the bin side, a transform, the ekind numbering or the unit, and explains the measurement-bundle error.
- **Refutes it:** the notes match my implementation. Then I will scan the remaining options empirically.

```python
print('numeric_note:', VOCAB['numeric_note'])
print('value_bin_note:', VOCAB['value_bin_note'])
print('event_kinds:', VOCAB['event_kinds'])
print('atom_kinds:', VOCAB['atom_kinds'])
print('codes:', VOCAB['codes'])
print('numeric example:', [ (d['code'],d['attribute'],{k_:v_ for k_,v_ in d.items() if k_ not in ('fine_bin_edges',)}) for d in VOCAB['numeric'] if d['code'] in ('map_arterial','glucose_finger')])
```