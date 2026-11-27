The normalisation is not under that key; the vocab's keys are config, codes, event_kinds, atom_kinds, numeric, numeric_note and the value-bin entries. It should be under 'numeric', so I'll print the note and the insulin and norepinephrine entries to see whether the dose z-score is linear or log-based.

```python
print(str(VOCAB['numeric_note'])[:400])
num=VOCAB['numeric']; print(type(num), list(num.keys())[:10] if isinstance(num,dict) else len(num))
for k in (num if isinstance(num,dict) else []):
    if 'insulin' in str(k) or 'norep' in str(k): print(k, str(num[k])[:300])
```