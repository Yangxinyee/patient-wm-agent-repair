The run failed inside the encoder replica, at `z32`, with a list being indexed by a tuple. One of the global names I reused in the last few runs probably shadows a global that `z32` depends on: `keep`, `fs`, `Us`, `at`, `bd`, `X` or `B`. I'll list the global names that `z32`, `atom_vectors` and `encode_bundles` use, and see which I have overwritten.

```python
for f in [z32, atom_vectors, encode_bundles]:
    c=f.__code__; print(f.__name__, c.co_names)
```