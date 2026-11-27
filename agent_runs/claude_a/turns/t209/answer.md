The internals pipeline is exact: hn recomputed from the stored hf and E matches `tf_loop` to 3e-5, and `n3_0` is available. So the gate and rate decoder can be fitted on the very inputs the block-0 FFN units will see.

The next ingredient is synthetic training data at doubled and halved insulin rates, so the decoder stays linear where the dose edits push it. For that I need to edit atoms and re-encode them with `encode_bundles`.

This run will:
- print the columns of `dsE['at']` and `dsE['bd']`;
- check that `encode_bundles(dsE['at'], dsE['bd'])` reproduces the stored e and u;
- count insulin and NE bundles.

If it reproduces them exactly, I can make edited copies of dsE (insulin ×2 and ×0.5 at every insulin bundle) and collect `n3_0` under them.

```python
at=dsE['at']; bd=dsE['bd']
print('at', type(at).__name__, list(at.columns) if hasattr(at,'columns') else at[:2])
print('bd', type(bd).__name__, list(bd.columns) if hasattr(bd,'columns') else bd[:2])
try:
    e2,u2=encode_bundles(at, bd)
    print('re-encode err e', float(np.abs(np.asarray(e2)-np.asarray(dsE['e'])).max()), 'u', float(np.abs(np.asarray(u2)-np.asarray(dsE['u'])).max()))
except Exception as ex: print('encode_bundles ERR', repr(ex)[:300])
print(at.head(3).to_string() if hasattr(at,'head') else '')
```