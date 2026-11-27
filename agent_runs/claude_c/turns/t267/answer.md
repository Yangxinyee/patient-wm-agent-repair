The codes are a list of dicts: insulin is code 8 (units/hour) and norepinephrine is code 9 (mcg/kg/min). `VOCAB['numeric']` is a list of 8 entries, so it is not indexed by code id. I'll print the entries whose text mentions either drug, to get each one's median, IQR and bin edges.

```python
for e in num:
    s=str(e)
    if 'insulin' in s or 'norep' in s or "'code': 8" in s or "'code': 9" in s: print(s[:600])
print(str(num[0])[:300])
```