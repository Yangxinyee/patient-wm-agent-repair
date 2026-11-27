The architecture shows how dose information enters. Insulin and norepinephrine atoms go through `encode_action.*` and produce u_k. u_k feeds the latent posterior `z_post([ctx, e_k, u_k])` and the prelude transformer as a token. A doubled dose changes the atom's z_dose/z_rate (through `embed.numeric`) and its dose/rate bins (`embed.bins.dose`, `embed.bins.rate`). The edit therefore reaches the state only through those embedding terms and the action encoder. That suggests a targeted place to edit: the embedding of drug atoms, or `encode_action`. Next I'll read the data README and the vocab config to learn the record layout and how doses are normalised. A layout with hourly infusion records plus glucose/MAP readings at irregular times would confirm my picture. It would also tell me how to estimate true drug effects from the records, e.g. whether the doses carry randomisation or exogenous variation.

```python
print(open('/data/README.md').read())
V = VOCAB
print(type(V), list(V.keys()) if isinstance(V, dict) else None)
for k in V:
    s = json.dumps(V[k])
    print(k, ":", s[:1500])
    print('---')
```