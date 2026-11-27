The factual CSV has, per reading: stay, hour, event, observed value, predicted mean and bin NLL. That is the target to check a local head against.

The vocab config has no list of head types or status fields. So the `type_emb` index (15 rows) and the 8 `last_numeric` inputs must be inferred, most likely from `event_kinds`, `atom_kinds` and the notes. I will print those entries briefly.

```python
for k in ['event_kinds','atom_kinds','numeric_note','value_bin_note']:
    print(k, '->', str(V[k])[:500])
print('codes:', str(V['codes'])[:400])
```