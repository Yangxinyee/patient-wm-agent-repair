The normalisation is linear: z = (x − median) / IQR per code, clipped to [−10, 10]. Fine bins count the edges below x. So the encoder's numeric inputs z_rate and z_dose scale linearly with the rate until clipping, which makes them a usable start for a linear channel.

`VOCAB['numeric']` is a list of 8 entries. I'll print the entries for the two drugs (median, IQR and edges for rate and dose) to see the z range real and edited rates cover and whether the ×2 or ×3 edits hit the ±10 clip.

```python
for e in VOCAB['numeric']:
    nm = str(e.get('code', e.get('name', '')) if isinstance(e, dict) else e)
    if isinstance(e, dict) and any(s in str(e) for s in ('insulin', 'norepinephrine')):
        print({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in e.items()})
```