The step formulas are as I summarised. The context uses `gap(log dt / 10)`, and the view's gap scaling is ambiguous, so I'll test both. Next I'll gather what the local head needs: the vocab keys (code → type), the embedding tensor names, and `last_numeric` column norms, to work out what its 8 inputs are.

```python
print(list(VOCAB.keys())[:30])
print('; '.join(f"{n}:{tuple(SHAPES[n])}" for n in NAMES if n.startswith('embed.') or 'last_status' in n or 'code_bias' in n))
W8 = npy(load_original('last_numeric.weight')); print('ln col norms', np.linalg.norm(W8, axis=0).round(3))
```