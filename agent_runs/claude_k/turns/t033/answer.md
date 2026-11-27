The readout is now pinned down. It uses the stay-hours whose record has the readout reading (glucose_finger or map_arterial), counted from just after the first dose from hour 12. On those exact rows, the original model is far too weak compared with my records-based truth:

| edit | model | records-based truth |
|---|---|---|
| insulin x2 | -2.15 | -13.8 |
| insulin x1/2 | +1.23 | +6.9 |
| norepinephrine x2 | +0.34 | +4.6 |
| norepinephrine x1/2 | -0.06 | -2.3 |

That is about 6× too weak for insulin and over 10× for norepinephrine. The model mostly uses doses as markers of the patient's state rather than as causes.

To design an edit I need the model's exact computation paths for drug atoms and for the glucose/MAP value heads. Next I'll list all 181 tensor names with shapes, grouped by module, to see the embedding tables, the action encoder, the core and the heads. I'm looking for a small set of tensors where a dose-dependent signal enters (embed.numeric's z_dose/z_rate columns, embed.bins.dose/rate, encode_action) and where the value head reads the state (attr_heads.value_bin, embed_transition). Those are the candidate edit points.

```python
T = json.load(open('/model/tensors.json'))
print(type(T), (list(T.keys())[:5] if isinstance(T, dict) else T[:3]))
for n in NAMES:
    print(n, SHAPES[n])
```