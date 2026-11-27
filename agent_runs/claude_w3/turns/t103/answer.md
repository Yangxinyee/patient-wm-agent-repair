The kernel did not keep its input history, so I can't read back the code that printed v1–v6. It did keep many helpers for a local copy of the model: atom_emb, build_bundles, bundle_vecs, set_enc, mlp, LN, the zpost_check functions, a `variants` dict with 7 entries and ENC_V='v1'. So v1–v6 were most likely z-posterior errors under different guesses about how bundles are encoded. All are around 0.08, except one that is badly wrong, so none matched exactly.

Next I will re-read ARCHITECTURE.md and try to recover the source of my helper functions, so I know what the local copy currently assumes. If the documentation shows a step I left out (an input feature, a normalisation, or a slot ordering), that would explain the leftover ~0.08 error.

```python
print(open('/model/ARCHITECTURE.md').read())
```