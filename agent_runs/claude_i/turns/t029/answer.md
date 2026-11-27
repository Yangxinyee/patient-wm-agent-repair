The z-score is (x − median)/IQR per code, clipped to ±10. Fine bins come from per-code edges.

The value-bin centres for finger glucose are coarse: 73, 84, 92, 103, 119, 137, 149, 160, 169, 178, 189, 204, 221, 236, 250, 272. A shift of −13 mg/dL in the predicted mean therefore means moving about 1–1.5 bins of probability mass.

Next I need the `numeric` list. It holds each code's median/IQR and its value, rate and dose bin edges, which show exactly how rate × c moves z_rate, z_dose and the bins for insulin and norepinephrine.

```python
num = VOCAB['numeric']
print(len(num))
for item in num:
    print(json.dumps(item)[:800])
```