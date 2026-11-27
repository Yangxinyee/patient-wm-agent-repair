The true one-step coefficients on the dose deviation are [−2.03, −0.94, 0.11, 0.55, 0.69, 0.46] (se 0.05). They match my earlier estimates closely.

I now need the model's coefficients on the identical specification. I can get them from the model's own predicted means, computed exactly with the local head from saved states.

I will collect states on training stays, keeping the tuning stays for validation, and start with 100 random stays that received insulin. The selection is saved first so the stay ids are reproducible. If the model's coefficients come out near zero (as the 38-stay estimate suggested), the missing effect is about the full true coefficient.

```python
rng = np.random.default_rng(123)
selA = sorted(rng.choice(ins_stays_tr, 100, replace=False).tolist())
ne_stays_tr = [s for s in stays_tr if (Gtr['ne'].loc[s].fillna(0)>0).any()]
selB = sorted(rng.choice([s for s in ne_stays_tr if s not in set(selA)], 100, replace=False).tolist())
rest = [s for s in stays_tr if s not in set(selA)|set(selB)]
selC = sorted(rng.choice(rest, 100, replace=False).tolist())
json.dump({'A': [int(s) for s in selA], 'B': [int(s) for s in selB], 'C': [int(s) for s in selC]}, open('/work/sel_train.json','w'))
print(json.dumps({"type": "states", "model": "original", "split": "train", "stays": [int(s) for s in selA]}))
```