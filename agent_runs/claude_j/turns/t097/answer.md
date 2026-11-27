ARCHITECTURE.md is short (5,234 characters, four sections), so I will re-read all of it now. The old replica functions have many option flags, such as unit indices, `kq`, aggregation (`'sqrt'`), memory mode and age mode. That suggests the previous session was searching over implementation details the document leaves open, in order to match the states output. I need the document's exact wording to judge which choices were forced and which were guesses.

After this I will write a clean replica into `/work/replica.py`. I will validate it against `/work/fwd_3/states_original.npz`, using the 5,116 bundles from 100 tuning stays with out, h and z. The replica is right only if it reproduces `out`, h and the z posterior mean to float precision, bundle by bundle.

```python
print(arch)
```