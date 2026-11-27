Each row is one stay × hour × edit, with the model's shift, its standard error over paths, the factual predicted mean and a "reached" flag. For stay 2020000000, hour 15 has a row but is not reached, while hours 16–24 are reached.

Next I will check against the tuning records:
1. Do rows exist exactly at hours with a recorded glucose_finger reading (insulin edits) or a map_arterial reading (NE edits)?
2. Does "reached" mean that a dose of the drug at hour 12 or later started before that reading?

Once the definition is pinned down I can compute my estimated true effects on exactly these stay-hours.

```python
Ptu2 = Ptu.copy()
s0 = 2020000000
print(Ptu2.loc[s0].loc[10:24].to_string())
for ed, code in [('insulin_x2','glucose_finger'),('ne_x2','map_arterial')]:
    x = de[de.edit==ed]
    # compare rows with reading presence
    rows = set(zip(x.stay_id, x.hour))
    pres = Ptu2[[code]].reset_index()
    pres = pres[(pres.h>=13)&(pres.h<=24)&pres[code].notna()]
    prs = set(zip(pres.stay_id, pres.h))
    stays_in = set(de.stay_id)
    prs = {r for r in prs if r[0] in stays_in}
    print(ed, len(rows), len(prs), len(rows & prs), len(rows - prs), len(prs - rows))
```