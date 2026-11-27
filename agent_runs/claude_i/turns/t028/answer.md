The tensor list shows how drug information enters. Each insulin or norepinephrine atom adds:
- `embed.numeric` columns for z_dose and z_rate (plus has_dose and has_rate);
- `embed.bins.dose` and `embed.bins.rate` rows for its fine bins.

These go through `encode_action` into the core.

Before choosing a lever I need the exact normalisation: are the z-scores computed on log or raw scale, with what centre and scale? I also need the rate/dose bin edges per code and the value-bin centres. Together these tell me how ×2, ×½, ×3 and ×0 move each feature, and how a shift in predicted glucose maps onto value bins.

```python
print(VOCAB['numeric_note']); print(VOCAB['value_bin_note'])
print(json.dumps(VOCAB['codes'])[:1500])
num = VOCAB['numeric']
print(type(num), (list(num.keys()) if isinstance(num, dict) else None))
for k in (num.keys() if isinstance(num, dict) else []):
    print(k, json.dumps(num[k])[:1500])
print(json.dumps(VOCAB['value_bin_centres'])[:3000])
print(VOCAB['event_kinds'], VOCAB['atom_kinds'])
```