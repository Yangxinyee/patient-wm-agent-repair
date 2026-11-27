Each row is one stay-hour, giving the shift, its SE, the factual predicted mean and a `reached` flag (the drug has run since hour 12). For stay 2020000000, insulin ×2 builds from −1.5 at hour 16 to about −3 by hours 21–22. Insulin ×½ is almost zero there, which suggests the model's response is asymmetric or that this stay's rates are small.

To see what a stay-hour is and how the model's effect evolves, I'll compare this stay's record with its rows. That shows whether rows exist only at finger-glucose hours or at all hours. I'll also summarise the model's shift by hours since the first edited dose and by the recorded rate.

```python
s0 = 2020000000
print(PT.loc[s0][['gf','gl','ma','mc','ins','ne']].T.round(2).to_string())
print("rows hours per stay (first 5 stays):")
for sid in de.stay_id.unique()[:5]:
    print(sid, sorted(de[(de.stay_id==sid)&(de.edit=='insulin_x2')].hour.tolist()), sorted(de[(de.stay_id==sid)&(de.edit=='ne_x2')].hour.tolist()))
print(de.groupby('edit').reached.agg(['sum','size']))
```