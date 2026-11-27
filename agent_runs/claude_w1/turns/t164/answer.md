head_mean failed because the global b1 now has shape (4,). The head replica reads globals W1, b1, W2, b2, TR and cid, and some later analysis overwrote b1. This is a bug in my kernel state, not in the model. I will reload the value-head weights from the model files and check the shapes. This run only lists the relevant tensor names and the current shapes of those globals, so the fix is exact. Nothing about the hypothesis changes.

```python
print([ (n, SHAPES[n]) for n in NAMES if 'value_bin' in n or 'transition' in n])
for nm in ['W1','b1','W2','b2','TR','cid']:
    v = globals().get(nm); print(nm, type(v).__name__, (np.shape(v) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else v)))
print('cid sample', list(cid.items())[:6] if isinstance(cid, dict) else '')
```