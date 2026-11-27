The dose z-score is linear: z = (x − median)/IQR per code, clipped to [−10, 10]. So doubling a high dose produces a large z jump, and with N5 that jump lands in the saturated region. `VOCAB['numeric']` is a list of 8 per-code entries. I'll print the insulin and norepinephrine entries (median, IQR and bin edges) to see how far doubled doses push z.

```python
codes=VOCAB['codes']; print(codes)
for i,c in enumerate(codes):
    if c in ('insulin','norepinephrine'): print(c, str(num[i])[:700])
```