My instrumented loop reproduces `tf_states_ds` exactly (max difference 0.0). With `ret=True`, `step_b` returns a tuple (y, out, dict of internals) rather than a dict, so the loop needs a small fix. Next I print the internals dict's keys and shapes to see what the prelude exposes: tokens, norm1 output, per-head attention weights, attention output, E, and the block/coda intermediates. From those I can build the staging decoder. My hypothesis is that `aw` is (batch, 4 heads, query tokens, 7 tokens) and `n1` is the normalised token matrix. That would let me compute head 3's attention-weighted value vectors per slot.

```python
dd_=res_[2]
for k_,v_ in dd_.items(): print(' ',k_, tuple(v_.shape) if hasattr(v_,'shape') else (type(v_).__name__, len(v_) if hasattr(v_,'__len__') else v_))
aw_=dd_.get('aw')
if aw_ is not None:
    print('aw sample (bundle0, per head, slot0 query -> tokens):'); print(np.round(aw_[0].numpy() if aw_.dim()==4 else aw_.numpy(),3)[...,0,:] if aw_.dim()==4 else np.round(aw_.numpy(),3))
print('bundle 1 rows info:'); print(ds_['bd'].iloc[rows0].to_string()[:800])
```