The function source can't be recovered (it was defined in the kernel without a file). I'll read each function's signature and the global names it references to work out what it does. I'll also print the tensor names grouped by module so I recall the weight layout. If `step` takes (h, M, out, bundle) and references prelude/blocks/coda weights, I'll call it on saved states to check accuracy. Otherwise I'll rewrite the core.

```python
for fn in ['step','step_err3','zcheck3','zpost_check2','prev_h','pred_out','enc_bundle','bundle_vecs','stay_bundles','ctx_nf','prelude_attn','mha','set_enc','HN']:
    f = globals().get(fn)
    if f is None: continue
    c = f.__code__
    print(fn, inspect.signature(f), '| names:', [n for n in c.co_names if n not in ('torch','np','F')][:25])
```