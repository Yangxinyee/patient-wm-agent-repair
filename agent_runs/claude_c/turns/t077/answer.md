Run 68 shows the vocab layout: the value-bin centres for glucose_finger and map_arterial, and a separate 'numeric' entry, which is where the drug normalisation and bin edges should be. I will print 'numeric' and its notes to get the insulin and norepinephrine z-score centre and scale and the rate and dose edges. Then I can compute how far the ×2 and ×½ edits move the tuning doses in z and bins, and how often they leave the training range.

```python
print(VOCAB['numeric_note'][:1500] if isinstance(VOCAB['numeric_note'],str) else VOCAB['numeric_note'])
print(VOCAB['value_bin_note'][:800] if isinstance(VOCAB['value_bin_note'],str) else VOCAB['value_bin_note'])
num=VOCAB['numeric']
print(type(num), (list(num.keys())[:30] if isinstance(num,dict) else len(num)))
import pprint
if isinstance(num,dict):
    for k_ in list(num.keys())[:12]:
        s_=json.dumps(num[k_]); print(k_, s_[:700])
else:
    for it in num[:12]: print(json.dumps(it)[:700])
```