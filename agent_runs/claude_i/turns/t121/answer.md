The core is now clear, and the config gives 4 attention heads.
- **Prelude:** one pre-norm transformer layer over 7 tokens (4 slots, e, u, z).
- **Recurrent part:** two layers, each with self-attention over the slots, cross-attention to a memory of past e vectors, and an FFN.
- **Coda:** one more transformer layer over the slots, applied after the linear map c_out.

Every bundle therefore passes the state through nonlinear residual layers. Any dose channel I add must survive repeated mixing.

So I cannot design the edit by reasoning on paper. I need a full offline reproduction of the core step, checked against the tool's states. With it I can roll the model forward under edited doses on CPU and test edits without spending forward passes.

My kernel already has the encoder (e, u, ctx, z_post), and fwd_3 holds the original model's states for 100 stays. Next I will list the signatures of my existing helpers and the shapes of the stored encodings, so I can build on them instead of rewriting.

```python
import inspect
for fn_name in ['build_bundles','encode_bundles','ctx_of','ctx_of2','eu_with_end','end_embed','atoms_embed','zpost_check','head_inputs','head_value2','LN','Wt','Tn','lnorm','actf','mlp_','lin_']:
    f_ = globals().get(fn_name)
    try: print(fn_name, inspect.signature(f_))
    except Exception as ex: print(fn_name, 'ERR', type(ex).__name__)
def shp(v_):
    if isinstance(v_, dict): return {k: shp(x) for k,x in list(v_.items())[:8]}
    return getattr(v_,'shape', type(v_).__name__)
for nm_ in ['euF','enc_','encB','eu','cfg','END_OPTS']:
    print(nm_, shp(globals()[nm_]))
print('st3', st3.shape, 'X3', X3.shape, 'S3', S3.shape, 'H3', H3.shape, 'NB', NB, 'zpF', tuple(zpF.shape), 'okm', okm.shape, okm.sum())
print('z3 keys', list(z3.keys()))
```