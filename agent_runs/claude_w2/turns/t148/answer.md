The factual output for the same 100 training stays (`fa4`) has, per reading, the tool's predicted mean, the NLL, and my earlier derived columns (row/prev bundle, dt, last value, age). The `mine` column, off by 8–13, was a failed head variant. Now that the states are exact, the head can be tested cleanly. It only needs a few definitions:
- which `type_emb` row is used;
- the 8 `last_numeric` inputs (ARCH names only 4);
- how `last_status` enters;
- which gap scaling the view uses.

I'll print the relevant vocab notes and the column/row norms of these tensors to narrow the hypotheses before testing.

```python
print(type(V['value_bin_centres']), (list(V['value_bin_centres'].keys())[:6] if isinstance(V['value_bin_centres'],dict) else np.array(V['value_bin_centres']).shape))
print('VBNOTE:', V['value_bin_note']); print('NUMNOTE:', V['numeric_note'])
print('EK:', V['event_kinds']); print('AK:', V['atom_kinds'])
print('codes:', [(c['name'],c['id']) for c in V['codes']])
print('last_numeric col norms', W['last_numeric.weight'].norm(dim=0).numpy().round(2))
print('last_status row norms', W['last_status.weight'].norm(dim=1).numpy().round(2))
print('type_emb row norms', W['type_emb.weight'].norm(dim=1).numpy().round(2))
print('assign',assign,'best_s',best_s)
```