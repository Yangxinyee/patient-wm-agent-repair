No helper computes the value head locally, so I'll write one and check it against the tool's factual predicted means. First I'll re-read the exact step formulas (lines 30–53) for the gap scaling and the posterior inputs.

```python
for i in range(28, 54):
    s = re.sub(r'(?i)insulin', 'D8', L_[i]); s = re.sub(r'(?i)norepinephrine', 'D9', s)
    print(i, s[:175])
```