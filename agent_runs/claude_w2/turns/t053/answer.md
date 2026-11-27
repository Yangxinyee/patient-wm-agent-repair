Factual output for the 100 training stays is in: 5,764 readings, 2,210 of them finger glucose. Original value NLL is 1.4235 per reading on these stays, versus 1.5279 on the tuning set. This is now the reference for matching my head re-implementation to the state rows from fwd_3.

Before writing the head I need four details:
- the layout of the factual CSV;
- how codes are indexed (`code_id`, VOCAB codes);
- which activations my `ACTS` dict provides;
- the format of the bin-centre table.

This is a short inspection run so the matching code is right first time.

```python
fa4 = pd.read_csv('/work/fwd_4/factual_original.csv')
print(fa4.columns.tolist()); print(fa4.head(4).to_string())
print('code_id', code_id); print('ek_id', ek_id); print('ACTS', list(ACTS.keys()))
print('codes', VOCAB['codes']); print('event_kinds', VOCAB['event_kinds'], 'atom_kinds', VOCAB['atom_kinds'])
vc_ = VOCAB['value_bin_centres']; print(type(vc_), (list(vc_.items())[:4] if isinstance(vc_, dict) else vc_[:3]))
print(list(W.keys())[:5], len(W), list(names.items())[:5] if isinstance(names, dict) else None)
print(S['hour'][:12], S['n_atoms'][:12])
```