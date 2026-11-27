The document is about 5.2k characters. The part that matters is the per-bundle step (lines 30–54), where the action token enters the state. I will print only that section.

```python
L=a.split('\n')
print('\n'.join(x[:160] for x in L[30:54]))
```