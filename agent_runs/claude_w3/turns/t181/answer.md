The z-posterior check gives about 10% relative error, so the encoder re-implementation is still inexact. Before deciding whether to rewrite it, I'll list the non-core tensors (embedding, encoders, heads) compactly. That will show which pieces, such as activation types, biases and the gap embedding, I need to get exactly right.

```python
rest=[n for n in NAMES if not n.startswith('core.')]
print(len(rest))
print(' '.join(f"{n}{list(SHAPES[n])}" for n in rest)[:5000])
```