The predicted mean is Σ p(b)·centre[b−1] over the 16 fine bins, and z-values are (x − median)/iqr clipped to ±10. The vocab config does not give `last_numeric`'s 8-input layout or the type index. I'll look at the per-code entries in the vocab for a type field.

```python
cd = VOCAB['codes']; print(type(cd).__name__, len(cd))
items = list(cd.items())[:13] if isinstance(cd, dict) else cd[:13]
for it in items:
    s = re.sub(r'(?i)insulin', 'D8', str(it)); s = re.sub(r'(?i)norepinephrine', 'D9', s)
    print(s[:150])
```