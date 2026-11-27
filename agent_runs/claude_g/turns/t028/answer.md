The relevant tensors are visible. Dose size enters drug atoms through four places: `embed.bins.dose` and `embed.bins.rate` (17 × 192 tables, row 0 = absent), and columns 2 (z_dose) and 4 (z_rate) of `embed.numeric.weight`. The categorical unit tables (`unit` with 6 rows, `rate_unit` with 4) are probably drug-specific and could absorb a per-drug constant.

Next I will read tensors.json, which may describe each tensor and its categories. I also want to confirm that the rate bins of the "last" register (used for readings' with_code) see row 0 for measurement codes. That would let me change rows 1–16 without touching reading predictions directly.

```python
for k,v in tj.items():
    if k.startswith('embed') or k in ['last_numeric.weight','last_status.weight','type_emb.weight']:
        print(k, json.dumps(v)[:600])
```