Run 212 supports the new route. The z surprise at a bundle tracks the bundle's glucose surprise closely: projection onto the fitted direction vG has correlation 0.76. For MAP the correlation is 0.58 along vM, which is nearly orthogonal to vG. The model's next prediction moves by 0.54 of a glucose surprise and 0.66 of a MAP surprise.

**Writer design:** two paired z_post hidden units per drug, gated to drug bundles. Their difference adds "innovation × direction" to the z mean. The innovation is the rate minus the rate predicted linearly from the bundle inputs (ctx, observations e). Building this needs e (the bundle's observation encoding), so I need my encoder helpers.

I'll check their signatures and whether my z posterior rebuild is accurate on drug bundles. If it is, I can fit the predicted rate directly on z_post's inputs.

```python
for f in ['setenc_my','setenc','view_from','mk','mk_best','base_mk','post_err','post_err2','post_err_g','zpost_err','enc_variant','zpost_variants','zerr_fn','ctx_best','u_rec','u_c']:
    o_=globals().get(f)
    if callable(o_):
        try: print(f, str(inspect.signature(o_))[:110])
        except Exception as e: print(f,'?')
    elif o_ is not None: print(f, type(o_).__name__)
print('U_EMPTY' in globals(), [k for k in globals() if 'EMPTY' in k or k.startswith('E_') or k.startswith('enc')][:10])
```