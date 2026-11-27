The helpers are there: `avec` builds an atom vector from its fields and `encset(vs, prefix)` encodes a set. Rebuilding insulin START atom fields by hand is error-prone, so I'll take a real one from `atoms_of_stay` and vary only its dose fields. First I'll print the record format for one insulin START atom and one NE START atom.

```python
ats = atoms_of_stay(grp_tr[s_any], RU)
print('type', type(ats).__name__, 'len', len(ats))
a0 = ats[0]; print('first atom type', type(a0).__name__, (list(a0.keys()) if isinstance(a0, dict) else str(a0)[:200]))
ins_at = [a for a in ats if (isinstance(a, dict) and 'insulin' in str(a.get('code', a.get('event', ''))) and 'END' not in str(a))][:1]
ne_at = [a for a in ats if (isinstance(a, dict) and 'norepinephrine' in str(a.get('code', a.get('event', ''))) and 'END' not in str(a))][:1]
print('insulin atom:', str(ins_at[0])[:400] if ins_at else None)
print('NE atom:', str(ne_at[0])[:400] if ne_at else None)
```