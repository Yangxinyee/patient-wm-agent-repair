All 8 `last_numeric` columns carry weight (norms 2.7–4.4), so the 8-input layout must be determined empirically. `last_status` row 0 is exactly zero, which looks like padding. The 15 `type_emb` rows may correspond to (event kind × atom kind).

Before writing the head I need the remaining pieces:
- the embedding table names and shapes;
- the code ids of all 13 codes (glucose_lab, map_arterial, insulin, norepinephrine);
- the layout of the per-code normalisation and bin information in VOCAB.

```python
print("; ".join(f"{n}{tuple(SHAPES[n])}" for n in NAMES if n.startswith('embed')))
print("codes:", ", ".join(f"{c['id']}:{c['name']}" for c in VOCAB['codes']))
nm0 = VOCAB['numeric']
print("numeric: len", len(nm0), "| item0 keys:", list(nm0[0].keys()) if isinstance(nm0[0],dict) else type(nm0[0]))
print("numeric item0:", str(nm0[0])[:400])
vbc0 = VOCAB['value_bin_centres']
print("value_bin_centres type:", type(vbc0).__name__, (list(vbc0.keys())[:6] if isinstance(vbc0,dict) else len(vbc0)))
```