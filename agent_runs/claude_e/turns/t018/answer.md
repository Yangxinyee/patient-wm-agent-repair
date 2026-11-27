The norepinephrine rate changes carry a lot of noise even when MAP is in range. In the 65–75 "hold" band, the 5–95% range of the log rate ratio is about −0.5 to +0.5. That gives plenty of hour-to-hour variation to identify the effect, provided the clinician's choices depend only on recorded values. Above 75 the rate is cut by about 15%; below 65 it is raised modestly.

Before going further into the records, I want a baseline of the model's current answers to the four dose questions. That tells me how large the discrepancy is and what the tool's output looks like. I'll choose tuning stays where both drugs run from hour 12 on, so that all four readouts carry information. This run only selects and saves those stays; the forward pass comes next.

```python
def drug_after12(G):
    i = G['ins'].loc[:, 12:24].notna().any(axis=1); n = G['ne'].loc[:, 12:24].notna().any(axis=1)
    return i, n
i_tu, n_tu = drug_after12(Gtu)
print('tuning stays: ins after 12:', i_tu.sum(), ' ne after 12:', n_tu.sum(), ' both:', (i_tu&n_tu).sum(), ' neither:', (~i_tu&~n_tu).sum())
both = list(Gtu['ins'].index[(i_tu&n_tu).values])
rest_i = list(Gtu['ins'].index[(i_tu&~n_tu).values]); rest_n = list(Gtu['ins'].index[(~i_tu&n_tu).values])
sel1 = (both + rest_i[:max(0,50-len(both)//2)] + rest_n)[:100]
print(len(both), len(sel1)); json.dump([int(s) for s in sel1], open('/work/sel1.json','w'))
print([int(s) for s in sel1])
```