The codes are: insulin 8, norepinephrine 9, glucose finger 6 and lab 10, MAP cuff 5 and arterial 7. The drug normalisation must sit under `numeric`, so I'll print the numeric note and the entries for codes 8 and 9: centre and scale of the robust z for rate and dose, and the bin edges. This tells me whether z_rate is linear in the rate or uses a log transform. That decides how a ×2 edit moves the encoder input and whether a linear channel can start from it.

```python
print(str(VOCAB['numeric_note'])[:400])
num = VOCAB['numeric']; print(type(num).__name__, (list(num.keys())[:12] if isinstance(num, dict) else len(num)))
for key in (list(num.keys()) if isinstance(num, dict) else []):
    if any(s in str(key) for s in ('8', '9', 'insulin', 'norepi')):
        print(key, '->', str(num[key])[:300])
```