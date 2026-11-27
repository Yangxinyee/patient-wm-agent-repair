The wrapper worked. `tf_states_ds` makes one batched `step_b` call per bundle position (54 calls for 10 stays) and returns the post-bundle h (540×4×192) and `out`. With `ret=True`, `step_b` returns a 3-tuple. The dataset holds:
- per-bundle e, u, time gaps and times;
- the bundle-row lists per stay;
- a readings table `rd` of 586 readings, with head inputs in `R`.

The evaluator's third output is the predicted mean per reading.

**Next step:** see what the 3-tuple contains. I expect the new h, `out`, and a dict of intermediates (E, x, `hn`, …). Then I can map E back to bundle rows through `srows`, which is all I need to fit the write decoder.

**What success looks like:** E of shape (n_stays, 4, 192) per call.

```python
c0 = CAP[0]
for i,x in enumerate(c0):
    if hasattr(x,'shape'): print(i, tuple(x.shape))
    elif isinstance(x,dict): print(i, 'dict', {k:(tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in x.items()})
    else: print(i, type(x))
print("srows[0][:5]", ds10['srows'][0][:5], " bd cols?", ds10['bd'].shape, ds10['bd'][:3])
print("R keys", {k:(tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in ds10['R'].items()})
print("rd sample", ds10['rd'][:3])
```