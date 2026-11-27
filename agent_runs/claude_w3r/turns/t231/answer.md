Each atom vector includes `embed.numeric @ [z_value, has_value, z_dose, has_dose, z_rate, has_rate]`, where z is a "robust per-code score". If z_rate is a linear function of the rate, the atom carries the rate linearly before mlp1. A dead-unit channel in mlp1 could then pass it on exactly. I will print `VOCAB['numeric']` and `VOCAB['numeric_note']` to see how z_rate is computed for insulin and norepinephrine.

```python
print(VOCAB['numeric_note']); print(json.dumps(VOCAB['numeric'])[:1500]); print(VOCAB['atom_kinds'], VOCAB['codes'])
```