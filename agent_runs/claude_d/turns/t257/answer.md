The tables are shared across drugs: `embed.bins.rate` and `embed.bins.dose` are (17, 192) and serve both insulin and norepinephrine. Their usage differs:
- Insulin trains the top bin 16 (0.7% of its records), and doubling sends 13% of insulin records there.
- Norepinephrine never uses bin 16, but 47% of its records land there when doubled.

So I cannot simply rewrite bin 16. That would disturb insulin's factual fit and its already-calibrated x2 response.

A norepinephrine-specific fix is possible inside `encode_action.mlp1`, the per-atom MLP:
- Use one or two hidden units as an AND-gate on "norepinephrine code + top rate/dose bin".
- Orient the gate orthogonally to the z-features so it is exactly constant on those atoms.
- Have it add a correction to the atom's mlp1 output, extending the trained norepinephrine representation linearly beyond rate 1.0.

No recorded atom ever opens this gate, so the factual cost should be essentially zero. Everything can be designed offline, because atom vectors and mlp1 are computable exactly from the embedding tables.

My earlier helper list includes `atom_vec` and `set_enc`, which may already implement this. I will check their signatures, find the categorical unit tables, and list the drug code entries.

```python
for f in ['atom_vec','set_enc','prep','step_of','mlp','LNf','lin','bundles_of','first_err']:
    try: print(f, inspect.signature(globals()[f]))
    except Exception as e: print(f, 'ERR')
print([n for n in NAMES if n.startswith('embed.') and 'bins' not in n and 'numeric' not in n], )
print([ (c['id'], c['name'], {k:v for k,v in c.items() if k not in ('id','name')}) for c in VOCAB['codes'] if c['name'] in ('insulin','norepinephrine')])
print(VOCAB['atom_kinds'], VOCAB['event_kinds'])
```